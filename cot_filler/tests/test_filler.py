import copy
import hashlib
import json
import fcntl
from pathlib import Path
import tempfile
import threading
import unittest
import subprocess
import sys
import urllib.error
import urllib.request
from unittest.mock import patch

from cot_filler.adapters import from_atif, from_messages, from_resolved_archive, from_session, load_traces
from cot_filler.core import canonical, digest, gaps_for, prompt_messages, validate_trace
from cot_filler.provider import ContextOverflow, FakeProvider, OpenAICompatibleProvider, check_budget
from cot_filler.server import make_server
from cot_filler.exporting import as_messages
from cot_filler.store import Conflict, Store
from cot_filler.masking import align_char_segments
from cot_filler.batch import generate_bulk
from cot_filler.deepseek_adapter import events_to_messages, render_events_with_segments
from cot_filler.jinja_renderer import DeepSeekJinjaRenderer


class _Encoding:
    ids = [10, 11, 12, 13]
    offsets = [(0, 2), (2, 5), (5, 6), (6, 9)]


class _OffsetTokenizer:
    def encode(self, text):
        return _Encoding()


class MaskTests(unittest.TestCase):
    def test_aligns_character_spans_to_tokens(self):
        result = align_char_segments("abcdefghi", [{"start_char": 0, "end_char": 5, "loss_intent": "mask"},
                                                     {"start_char": 5, "end_char": 9, "loss_intent": "train"}], _OffsetTokenizer())
        self.assertEqual(result["loss_mask"], [0, 0, 1, 1])

    def test_rejects_missing_offsets(self):
        with self.assertRaises(ValueError):
            align_char_segments("abc", [{"start_char": 0, "end_char": 3, "loss_intent": "train"}], object())


class DeepSeekAdapterTests(unittest.TestCase):
    class Encoder:
        @staticmethod
        def encode_messages(messages, **kwargs):
            return "\n".join(m["content"] for m in messages)

    def test_renders_rationale_and_finds_spans(self):
        rendered, spans = render_events_with_segments([
            {"event_id": "u", "kind": "message", "role": "user", "content": "Do it"},
            {"event_id": "r", "kind": "synthetic_rationale", "role": "assistant", "content": "Need to act", "before_event_id": "a"},
            {"event_id": "a", "kind": "message", "role": "assistant", "content": "Done"},
        ], self.Encoder)
        self.assertIn("<cot>\nNeed to act\n</cot>", rendered)
        self.assertEqual([s["loss_intent"] for s in spans], ["mask", "mask", "train"])

    def test_maps_calls_and_results_with_json_arguments(self):
        messages = events_to_messages([
            {"kind": "message", "role": "user", "content": "Run it"},
            {"kind": "tool_call", "role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "run", "arguments": {"x": 1}}}]},
            {"kind": "tool_result", "role": "tool", "tool_call_id": "c1", "content": "ok"},
        ])
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], '{"x":1}')
        self.assertEqual(messages[2]["tool_call_id"], "c1")

    def test_rejects_ambiguous_tool_definition(self):
        with self.assertRaises(ValueError):
            events_to_messages([{ "kind": "tool_definition", "role": "system", "data": {"name": "run"}}])

    def test_attaches_complete_definition_to_next_user_message(self):
        messages = events_to_messages([
            {"kind": "tool_definition", "role": "system", "data": {"name": "run", "description": "Run", "parameters": {"type": "object"}}},
            {"kind": "message", "role": "user", "content": "Run it"},
        ])
        self.assertEqual(messages[0]["tools"][0]["function"]["name"], "run")

    def test_upstream_jinja_matches_native_demo(self):
        root = Path(__file__).parents[1] / "assets" / "deepseek-v4-flash-0731"
        renderer = DeepSeekJinjaRenderer(root / "chat_template.jinja")
        events = json.loads((root.parent.parent / "examples" / "demo.json").read_text())["events"]
        messages = events_to_messages(events)
        jinja_messages = renderer._jinja_messages(messages)
        rendered = renderer.render(messages)
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_dsv4", root / "encoding" / "encoding_dsv4.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        native = module.encode_messages(messages, thinking_mode="thinking", drop_thinking=True)
        self.assertEqual(rendered, native)
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.assertEqual(tokenizer.encode(rendered).ids, tokenizer.encode(native).ids)

DEMO = Path(__file__).parents[1] / "examples/demo.json"


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.trace = json.loads(DEMO.read_text())

    def test_exact_independent_windows_and_tool_preservation(self):
        original = copy.deepcopy(self.trace)
        gaps = list(gaps_for(self.trace))
        self.assertEqual(gaps[0]["lookahead"], self.trace["events"][1:6])
        self.assertEqual(gaps[1]["prefix"], self.trace["events"][:3])
        self.assertEqual(len(gaps[2]["lookahead"]), 3)
        self.assertEqual(gaps[0]["target"]["tool_calls"][0]["function"]["arguments"]["thinking"], False)
        gaps[0]["prefix"][0]["content"] = "generated contamination"
        self.assertEqual(self.trace, original)
        self.assertNotIn("generated contamination", canonical(list(gaps_for(self.trace))[1]))

    def test_no_silent_long_context_crop(self):
        self.trace["events"][0]["content"] = "BEGIN" + "x" * 100000 + "END"
        prompt = prompt_messages(next(gaps_for(self.trace)))[1]["content"]
        self.assertIn("BEGIN" + "x" * 100000 + "END", prompt)
        with self.assertRaises(ContextOverflow):
            check_budget(12000, 1000, 13000)
        self.assertEqual(check_budget(12000, 1000, 14000), 13256)

    def test_teacher_blob_omits_tool_schemas_but_keeps_calls(self):
        trace = copy.deepcopy(self.trace)
        trace["events"].insert(1, {"event_id": "def", "kind": "tool_definition", "role": "system", "content": "SECRET SCHEMA"})
        gap = next(gaps_for(trace))
        prompt = prompt_messages(gap)[1]["content"]
        self.assertNotIn("SECRET SCHEMA", prompt)
        self.assertIn("tool_calls", prompt)

    def test_unresolved_duplicate_reasoning_and_synthetic_rejected(self):
        for mutate in [lambda t:t["events"][0].update(content_ref="rocks:42"),
                       lambda t:t["events"][1].update(event_id="u1"),
                       lambda t:t["events"][1].update(reasoning_content="private"),
                       lambda t:t["events"][1].update(kind="synthetic_rationale", synthetic=True),
                       lambda t:t["metadata"].update(synthetic_reasoning=True)]:
            trace = copy.deepcopy(self.trace)
            mutate(trace)
            with self.assertRaises(ValueError):
                list(gaps_for(trace))

    def test_gaps_only_assistant(self):
        self.trace["gap_targets"] = [{"event_id": "t1"}]
        with self.assertRaises(ValueError):
            list(gaps_for(self.trace))

    def test_atif_tool_payload_and_plaintext_presence(self):
        tool_result = {"type": "reasoning", "value": {"thinking": False, "encrypted_content": "ordinary field"}}
        doc = {"session_id": "x", "steps": [{"source":"user","message":"Task"},
            {"source":"agent","step_id":2,"message":"", "extra":{"thinking":"[encrypted reasoning]"},
             "tool_calls":[{"arguments":{"thinking":False,"reasoning":"task data"}}],
             "observation":{"duration":1,"results":[{"content":tool_result}]}},
            {"source":"agent","message":"Finished", "extra":{"thinking":"original private text"}}]}
        trace = validate_trace(from_atif(doc,"x"))
        self.assertEqual(trace["events"][2]["result"]["content"],tool_result)
        self.assertEqual(trace["events"][2]["observation_metadata"],{"duration":1})
        self.assertNotIn("original private text", canonical(trace))
        self.assertEqual(len(list(gaps_for(trace,"missing-assistant"))),1)
        self.assertEqual(len(list(gaps_for(trace))),1)

    def test_session_marker_maps_to_tool_action_and_never_floats(self):
        raw=[{"type":"user_message","text":"Task"},
             {"type":"response_item","payload":{"type":"reasoning","encrypted_content":"secret"}},
             {"type":"response_item","payload":{"type":"function_call","id":"call","name":"run","arguments":{"reasoning":"normal payload"}}}]
        trace=validate_trace(from_session(raw,"s"))
        gap=next(gaps_for(trace))
        self.assertEqual(gap["target"]["event_id"],"call")
        self.assertNotIn("secret",canonical(trace))
        raw[2]["payload"]={"type":"function_call_output","output":"later"}
        with self.assertRaises(ValueError):
            from_session(raw,"s")

    def test_embedded_original_reasoning_presence_across_adapters(self):
        session=from_session([{"type":"user_message","text":"Task"},{"type":"response_item","payload":{"type":"message","role":"assistant","content":"Answer","reasoning_content":"private placeholder"}}],"s")
        messages=from_messages({"messages":[{"role":"user","content":"Task"},{"role":"assistant","content":[{"type":"thinking","thinking":"private placeholder"},{"type":"text","text":"Answer"}]}]},"m")
        for trace in (session,messages):
            self.assertEqual(list(gaps_for(trace,"missing-assistant")),[])
            self.assertNotIn("private placeholder",canonical(trace))

    def test_session_synthetic_provenance_rejected_before_wrapping(self):
        raw=[{"type":"response_item","payload":{"type":"message","role":"assistant","content":"Generated placeholder","synthetic":True}},
             {"type":"response_item","payload":{"type":"reasoning","encrypted_content":"marker"}},
             {"type":"assistant_message","text":"Answer"}]
        with self.assertRaises(ValueError):
            from_session(raw,"s")

    def test_resolved_archive_keeps_structured_data_and_maps_markers(self):
        call={"type":"function_call","call_id":"call1","name":"inspect","arguments":{"reasoning":"ordinary tool argument","thinking":False}}
        doc={"resolved":True,"archive_events":[
            {"event_id":"u","role":"user","kind":"message","content":"Task","data":{"role":"user","content":"Task"}},
            {"event_id":"r","role":"assistant","kind":"reasoning","content":"private marker","data":{"type":"reasoning","encrypted_content":"marker"}},
            {"event_id":"a","role":"assistant","kind":"tool_call","content":"rendered arguments","data":{"block":call}},
            {"event_id":"t","role":"tool","kind":"tool_result","content":"output","data":{"call_id":"call1","output":"output"}}]}
        trace=validate_trace(from_resolved_archive(doc,"archive"))
        self.assertEqual(trace["events"][1]["data"]["block"],call)
        self.assertEqual(next(gaps_for(trace))["event_id"],"a")
        self.assertNotIn("private marker",canonical(trace))
        doc["resolved"]=False
        with self.assertRaises(ValueError): from_resolved_archive(doc,"archive")

    def test_archive_embedded_reasoning_is_removed_or_rejected(self):
        doc={"resolved":True,"archive_events":[{"event_id":"a","role":"assistant","kind":"message","content":"raw","data":{"block":{"role":"assistant","content":[{"type":"thinking","thinking":"private placeholder"},{"type":"text","text":"Visible"}]}}}]}
        trace=validate_trace(from_resolved_archive(doc,"archive"))
        self.assertEqual(trace["events"][0]["content"],"Visible")
        self.assertNotIn("private placeholder",canonical(trace))
        doc["archive_events"][0]["data"]["block"]["content"]="<think>private placeholder</think>Visible"
        with self.assertRaises(ValueError): validate_trace(from_resolved_archive(doc,"archive"))

    def test_archive_string_block_and_message_fields_are_supported(self):
        fields={"role":"assistant","id":"source-a","recipient":"all","reasoning_summary":"private placeholder"}
        doc={"resolved":True,"archive_events":[{"event_id":"a","role":"assistant","kind":"message","content":"Visible answer","data":{"block":"Visible answer","message_fields":fields}}]}
        trace=validate_trace(from_resolved_archive(doc,"archive"))
        self.assertEqual(trace["events"][0]["data"]["block"],"Visible answer")
        self.assertEqual(trace["events"][0]["data"]["message_fields"],{"role":"assistant","id":"source-a","recipient":"all"})
        self.assertNotIn("private placeholder",canonical(trace))
        self.assertEqual(list(gaps_for(trace,"missing-assistant")),[])
        doc["archive_events"][0]["data"]["block"]="<think>private placeholder</think>Visible answer"
        with self.assertRaises(ValueError): validate_trace(from_resolved_archive(doc,"archive"))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"review.sqlite3"
        self.source=Path(self.tmp.name)/"source.json"
        self.source.write_bytes(DEMO.read_bytes())
        self.store=Store(self.path)
        trace, provenance=next(load_traces(self.source))
        self.store.import_trace(trace,provenance)
        self.gap=self.store.list_gaps()[0]["id"]
        text,data=FakeProvider().generate(self.store.detail(self.gap)["gap"])
        self.store.add_candidate(self.gap,text,data,0)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def external_record(self, gap_id=None):
        detail=self.store.detail(gap_id or self.store.list_gaps()[1]["id"])
        gap=detail["gap"]
        return {"schema":"cot.external-candidate.v1", "gap_id":gap["id"],
                **{key:gap[key] for key in ("source_digest","prefix_hash","lookahead_hash","prompt_hash")},
                "expected_revision":detail["candidate"]["revision"] if detail["candidate"] else 0,
                "text":"The observed initial value explains the extra unit, so change that initializer and test the result.",
                "generator":{"provider":"codex-session","model":"not_attested","authoring_task":"/root/test_author","prompt_version":gap["prompt_version"],"token_usage":None}}

    def test_review_durable_and_revisions_keep_original(self):
        self.store.review(self.gap,1,"approved","Specific reviewed explanation.","fits",["edited"])
        original=self.store.detail(self.gap)["candidate"]["original_text"]
        self.store.close()
        self.store=Store(self.path)
        self.assertEqual(self.store.detail(self.gap)["candidate"]["text"],"Specific reviewed explanation.")
        self.assertIn("DEMO",original)
        self.store.request_regeneration(self.gap,1)
        self.assertEqual(list(self.store.export()),[])
        self.store.add_candidate(self.gap,"A newer explanation.",{},1)
        with self.assertRaises(Conflict):
            self.store.review(self.gap,1,"approved")
        self.assertEqual(self.store.detail(self.gap)["candidate"]["status"],"pending")
        self.assertEqual(list(self.store.export()),[])
        self.assertEqual(len(self.store.detail(self.gap)["history"]),2)

    def test_three_arms_preserve_source_and_precede_action(self):
        self.store.review(self.gap,1,"approved","Inspect the implementation to locate its initial-value error.")
        arms={a:list(self.store.export(a))[0] for a in ("visible","masked","none")}
        trace=json.loads(self.source.read_text())
        self.assertEqual(arms["none"]["events"],trace["events"])
        self.assertEqual([e for e in arms["visible"]["events"] if e["kind"]!="synthetic_rationale"],trace["events"])
        self.assertEqual(arms["visible"]["events"][1]["kind"],"synthetic_rationale")
        self.assertEqual(arms["visible"]["events"][2],trace["events"][1])
        self.assertEqual(arms["masked"]["metadata"]["loss_segments"][1]["loss_intent"],"mask")
        self.assertFalse(arms["visible"]["metadata"]["training_ready"])
        self.assertEqual(arms["none"]["metadata"]["approved_gaps"],arms["visible"]["metadata"]["approved_gaps"])

    def test_messages_export_preserves_tools_and_rationale_before_action(self):
        self.store.review(self.gap,1,"approved","Inspect the implementation before choosing a change.")
        row=as_messages(list(self.store.export("masked"))[0])
        original=json.loads(self.source.read_text())["events"]
        self.assertEqual(row["metadata"]["original_events"],original)
        self.assertTrue(row["messages"][1]["content"].startswith("<cot>\nInspect"))
        self.assertEqual(row["messages"][1]["tool_calls"],original[1]["tool_calls"])
        self.assertEqual(row["messages"][2]["tool_call_id"],"read-1")
        span=row["metadata"]["message_content_segments"][1]
        self.assertEqual(span["loss_intent"],"mask")
        self.assertFalse(row["metadata"]["training_ready"])
        canonical_row=list(self.store.export())[0]
        canonical_row["events"][0]["kind"]="tool_definition"
        with self.assertRaises(ValueError): as_messages(canonical_row)

    def test_source_changed_blocks_generation_review_export(self):
        self.store.review(self.gap,1,"approved","Reviewed.")
        self.source.write_text("{}")
        with self.assertRaises(Conflict): self.store.verify_source(self.gap)
        with self.assertRaises(Conflict): self.store.review(self.gap,1,"rejected")
        with self.assertRaises(Conflict): list(self.store.export())

    def test_tampered_gap_is_rejected(self):
        gap=self.store.detail(self.gap)["gap"]
        gap["prefix"][0]["content"]="different branch"
        with self.store.db:
            self.store.db.execute("UPDATE gaps SET data=? WHERE id=?",(canonical(gap),self.gap))
        with self.assertRaises(Conflict): self.store.verify_source(self.gap)

    def test_validation_blocks_approval(self):
        self.store.add_candidate(self.gap,"A partial explanation",{"flags":["truncated_output"]},1)
        with self.assertRaises(ValueError): self.store.review(self.gap,2,"approved")
        with self.assertRaises(ValueError): self.store.review(self.gap,2,"approved","<cot>malformed</cot>")
        self.store.review(self.gap,2,"approved","A complete edited explanation.")

    def test_external_import_keeps_provenance_and_requires_manual_review(self):
        record=self.external_record()
        before=self.source.read_bytes()
        result=self.store.import_candidates([record],{"source":"test_finished_artifact"})
        self.assertEqual(result,[{"gap_id":record["gap_id"],"revision":1,"status":"pending"}])
        candidate=self.store.detail(record["gap_id"])["candidate"]
        self.assertEqual(candidate["generator"],record["generator"])
        self.assertEqual(candidate["completion_status"],"assistant_authored_finished_artifact")
        self.assertNotIn("finish_reason",candidate)
        self.assertIn("semantic_review_required",candidate["flags"])
        self.assertEqual(candidate["external_record_sha256"],digest(record))
        self.assertEqual(list(self.store.export()),[])
        self.assertEqual(self.source.read_bytes(),before)

    def test_external_import_rejects_mismatch_stale_and_false_provenance(self):
        valid=self.external_record()
        variations=[]
        for field in ("source_digest","prefix_hash","lookahead_hash","prompt_hash"):
            bad=copy.deepcopy(valid);bad[field]="0"*64;variations.append(bad)
        bad=copy.deepcopy(valid);bad["generator"]["model"]="DeepSeek-V4-Flash";variations.append(bad)
        bad=copy.deepcopy(valid);bad["generator"]["token_usage"]=100;variations.append(bad)
        bad=copy.deepcopy(valid);bad["generator"]["prompt_version"]="wrong";variations.append(bad)
        bad=copy.deepcopy(valid);bad["status"]="approved";variations.append(bad)
        bad=copy.deepcopy(valid);bad["expected_revision"]=False;variations.append(bad)
        bad=copy.deepcopy(valid);bad["text"]="<cot>Malformed wrapper</cot>";variations.append(bad)
        for bad in variations:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):self.store.import_candidates([bad])
                self.assertIsNone(self.store.detail(valid["gap_id"])["candidate"])
        self.store.import_candidates([valid])
        with self.assertRaises(Conflict):self.store.import_candidates([valid])
        self.assertEqual(len(self.store.detail(valid["gap_id"])["history"]),1)

    def test_external_batch_is_atomic_on_late_error_and_duplicates(self):
        first=self.external_record()
        second=self.external_record(self.store.list_gaps()[2]["id"])
        second["prompt_hash"]="incorrect"
        with self.assertRaises(ValueError):self.store.import_candidates([first,second])
        self.assertIsNone(self.store.detail(first["gap_id"])["candidate"])
        with self.assertRaises(ValueError):self.store.import_candidates([first,first])
        second=self.external_record();second["gap_id"]="unknown"
        with self.assertRaises(KeyError):self.store.import_candidates([first,second])
        self.assertIsNone(self.store.detail(first["gap_id"])["candidate"])

    def test_external_import_rejects_changed_source_file(self):
        record=self.external_record()
        self.source.write_text("{}")
        with self.assertRaises(Conflict):self.store.import_candidates([record])
        self.assertIsNone(self.store.detail(record["gap_id"])["candidate"])

    def test_api_session_origin_stale_revision_and_regeneration(self):
        server=make_server(self.path,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        base=f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base+"/api/session") as r: token=json.load(r)["token"]
            payload=json.dumps({"revision":1,"status":"approved","text":"Reviewed explanation."}).encode()
            req=urllib.request.Request(base+f"/api/gaps/{self.gap}/review",data=payload,headers={"Content-Type":"application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught: urllib.request.urlopen(req)
            self.assertEqual(caught.exception.code,403)
            req.add_header("X-Cot-Token",token)
            req.add_header("Origin","https://untrusted.example")
            with self.assertRaises(urllib.error.HTTPError): urllib.request.urlopen(req)
            req.remove_header("Origin")
            with urllib.request.urlopen(req) as response: self.assertEqual(json.load(response)["candidate"]["status"],"approved")
            record=self.external_record()
            req=urllib.request.Request(base+"/api/candidates/import",data=json.dumps({"candidates":[record]}).encode(),headers={"Content-Type":"application/json","X-Cot-Token":token})
            with urllib.request.urlopen(req) as response:
                imported=json.load(response)
                self.assertEqual(imported["candidates"][0]["status"],"pending")
                self.assertFalse(imported["automatic_approval"])
                self.assertFalse(imported["inference_started"])
            with urllib.request.urlopen(base+"/") as response:
                html=response.read().decode()
                self.assertIn("textContent",html)
                self.assertNotIn("innerHTML",html)
        finally:
            server.shutdown();server.server_close();thread.join()


class ProviderTests(unittest.TestCase):
    def config(self):
        return {"model":"test","base_url":"http://127.0.0.1:9999/v1","tokenizer_path":"unused","context_limit":4096,"max_output_tokens":1024,"chat_template_matches_server":True,"chat_template_kwargs":{"enable_thinking":True}}

    def test_tokenizer_gets_identical_template_options_before_request(self):
        class Tokenizer:
            identity="test"
            def count(self,messages,options):
                self.options=options
                return 4000 if options["enable_thinking"] else 1
        tokenizer=Tokenizer()
        provider=OpenAICompatibleProvider(self.config(),tokenizer)
        gap=next(gaps_for(json.loads(DEMO.read_text())))
        with patch("urllib.request.build_opener") as network:
            with self.assertRaises(ContextOverflow): provider.generate(gap)
            network.assert_not_called()
        self.assertEqual(tokenizer.options,{"enable_thinking":True})

    def test_finish_length_flagged_without_using_hidden_response_reasoning(self):
        class Tokenizer:
            identity="test"
            def count(self,messages,options): return 10
        class Response:
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def read(self,limit): return json.dumps({"choices":[{"message":{"content":"Proposed explanation", "reasoning_content":"private provider reasoning"},"finish_reason":"length"}]}).encode()
        provider=OpenAICompatibleProvider(self.config(),Tokenizer())
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value=Response()
            text,data=provider.generate(next(gaps_for(json.loads(DEMO.read_text()))))
            request=opener.return_value.open.call_args.args[0]
            body=json.loads(request.data)
        self.assertIn("truncated_output",data["flags"])
        self.assertNotIn("private provider reasoning",canonical(data))
        self.assertEqual(body["chat_template_kwargs"],data["counted_chat_template_kwargs"])


class CommandTests(unittest.TestCase):
    def test_fake_cli_and_export_protect_original_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=str(Path(tmp)/"review.sqlite3")
            base=[sys.executable,"-m","cot_filler","--db",db]
            result=subprocess.run(base+["import",str(DEMO)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run(base+["generate","--fake","--limit","2"],capture_output=True,text=True)
            self.assertEqual(json.loads(result.stdout)["generated"],2)
            result=subprocess.run(base+["export","--output",str(DEMO)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("cannot overwrite",result.stderr)

    def test_occupied_inference_lock_prevents_real_provider_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path=Path(tmp)/"inference.lock"
            lock_path.touch()
            with lock_path.open("r+") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=subprocess.run([sys.executable,"-m","cot_filler","--db",str(Path(tmp)/"sidecar.sqlite3"),"generate","--config","missing-config-must-not-be-read.json","--inference-lock",str(lock_path)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("reserved by another client",result.stderr)

    def test_external_import_cli_preserves_pending_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=str(Path(tmp)/"review.sqlite3")
            base=[sys.executable,"-B","-m","cot_filler","--db",db]
            subprocess.run(base+["import",str(DEMO)],check=True,capture_output=True)
            store=Store(db)
            gap=store.detail(store.list_gaps()[0]["id"])["gap"]
            record={"schema":"cot.external-candidate.v1","gap_id":gap["id"],**{k:gap[k] for k in ("source_digest","prefix_hash","lookahead_hash","prompt_hash")},"expected_revision":0,"text":"Inspect the function before making a change, following the visible request.","generator":{"provider":"codex-session","model":"not_attested","authoring_task":"/root/cli_author","prompt_version":gap["prompt_version"],"token_usage":None}}
            path=Path(tmp)/"finished.jsonl"
            path.write_text(canonical(record)+"\n")
            result=subprocess.run(base+["import-candidates",str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)["imported"],1)
            self.assertEqual(store.detail(gap["id"])["candidate"]["status"],"pending")
            self.assertEqual(list(store.export()),[])
            store.close()


if __name__ == "__main__": unittest.main()
