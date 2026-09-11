"""Official native reasoning placement, omitting only conflicting generated gaps.

Original source/generation receipts are immutable. A omitted insertion is not a
rejection of its trace or a semantic review of the candidate.
"""
from copy import deepcopy
from collections import Counter
import hashlib
from pathlib import Path

from cot_filler.core import digest,validate_trace
from training.qwen38_no_cot.render import prefix_token_cutoff,training_row as base_training_row
from training.qwen38_turn_boundary_v2 import masked as native_parent
from training.qwen38_turn_boundary_v2 import source_adapters as baseline_source_adapters
from . import normalize,source_adapters

VERSION='qwen3.8-xhigh-native-leading-gap-cot-input-only/v4'
MASK_POLICY='assistant_content_and_eos_only_native_gap_cot_masked_v4'
OMISSION_POLICY='keep-native-leading-gap-omit-only-conflicting-cot/v4'
CROSS_ARM_PARITY_SCHEMA='qwen38-cross-arm-no-cot-parity/v2'
MaskedBoundaryExclusion=native_parent.MaskedBoundaryExclusion


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CrossArmParityError(ValueError):
    """The masked converter changed a baseline-visible target or source projection."""
    def __init__(self,audit):
        self.audit=deepcopy(audit)
        super().__init__('cross_arm_no_cot_parity_failed')


def _parity_stage_error(stage,error):
    return CrossArmParityError({'schema':CROSS_ARM_PARITY_SCHEMA,'verified':False,
        'adapter_divergences':[{'kind':'parity_evaluation_error','stage':stage,
            'error_type':type(error).__name__}]})


def _active_ids(result):
    return [token for token,label in zip(result['input_ids'],result['labels']) if label!=-100]


def _first_difference(left,right):
    for index,(a,b) in enumerate(zip(left,right)):
        if a!=b:return index
    return min(len(left),len(right)) if len(left)!=len(right) else None


def _sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prune_conflicting_reasoning(messages,boundaries):
    """Use the exact frozen native merger to identify, never relocate, a gap."""
    working=deepcopy(messages);omitted=[]
    for _ in range(1+sum(bool(m.get('reasoning_content')) for m in working)):
        try:
            normalize.coalesce_assistant_continuations(working,boundaries=boundaries,reasoning_policy='preserve')
            return working,omitted
        except normalize.ContinuationMappingError as error:
            if error.code not in {'reasoning_after_visible_content','multiple_reasoning_blocks'}:raise
            failures=error.audit.get('exclusions',[])
            if len(failures)!=1:raise ValueError('Ambiguous native reasoning conflict') from error
            index=failures[0].get('input_index')
            if type(index)is not int or not 0<=index<len(working) or not working[index].get('reasoning_content'):
                raise ValueError('Native conflict does not identify a filled original fragment') from error
            text=working[index].pop('reasoning_content')
            omitted.append({'original_message_index':index,'reason':error.code,'candidate_sha256':digest(text)})
    raise ValueError('Native gap pruning did not converge')


class Qwen38TurnBoundaryMaskedRenderer(native_parent.Qwen38TurnBoundaryMaskedRenderer):
    def __init__(self,tokenizer_dir,*,max_tokens=262144):
        super().__init__(tokenizer_dir,max_tokens=max_tokens)
        self.identity.update(version=VERSION,mask_policy=MASK_POLICY,renderer_sha256=_sha(__file__),
            native_parent_renderer_sha256=_sha(native_parent.__file__),
            turn_boundary_normalizer_sha256=_sha(normalize.__file__),
            turn_boundary_source_adapter_sha256=_sha(source_adapters.__file__),
            cross_arm_baseline_source_adapter_sha256=_sha(baseline_source_adapters.__file__),
            cross_arm_parity_schema=CROSS_ARM_PARITY_SCHEMA,
            reasoning_placement_policy=OMISSION_POLICY,omitted_gap_policy=OMISSION_POLICY,
            inline_reasoning_inserted=False,ordinary_native_assistant_template=True)
        self.identity['adaptations']=[x for x in self.identity['adaptations'] if x!='exclude_whole_source_if_filled_reasoning_would_move']+['omit_only_conflicting_generated_gaps_retain_visible_trace']

    def _rendered_input_difference_audit(self, zero_cot, actual, retained_texts):
        """Prove that filled input differs only by exact receipt-bound CoT text."""
        projected = actual['full_rendered_text']
        segment_hashes = []
        located = 0
        for index, text in enumerate(retained_texts):
            edits = []
            # The pinned official template applies Jinja's ``trim`` filter to
            # reasoning_content before placing it in the native think block.
            escaped = self._baseline._escape(text, f'/parity/reasoning/{index}', edits).strip()
            filled = '<think>\n' + escaped + '\n</think>\n\n'
            empty = '<think>\n\n</think>\n\n'
            position = projected.find(filled)
            if position < 0:
                break
            projected = projected[:position] + empty + projected[position + len(filled):]
            segment_hashes.append(_sha_text(filled))
            located += 1
        ranges = actual['cot_mask_audit']['retained_token_ranges']
        retained_ids = [actual['input_ids'][position]
                        for start, end in ranges for position in range(start, end)]
        return {
            'schema': 'qwen38-filled-input-difference-audit/v1',
            'verified': (located == len(retained_texts)
                         and projected == zero_cot['full_rendered_text']),
            'expected_reasoning_count': len(retained_texts),
            'located_reasoning_count': located,
            'zero_cot_rendered_sha256': _sha_text(zero_cot['full_rendered_text']),
            'actual_rendered_sha256': _sha_text(actual['full_rendered_text']),
            'stripped_actual_rendered_sha256': _sha_text(projected),
            'filled_rendered_segment_sha256': segment_hashes,
            'retained_reasoning_token_ranges': deepcopy(ranges),
            'retained_reasoning_token_count': len(retained_ids),
            'retained_reasoning_token_ids_sha256': digest(retained_ids),
            'difference_policy': 'replace-exact-receipt-bound-filled-think-with-empty-think/v1',
        }

    def _cross_arm_audit(self,baseline_messages,baseline_boundary_audit,zero_cot,actual,*,
                         require_exact_zero,retained_texts=()):
        baseline=prefix_token_cutoff(self._baseline.render(baseline_messages,None),max_tokens=self.max_tokens)
        baseline_targets,zero_targets,actual_targets=map(_active_ids,(baseline,zero_cot,actual))
        violations=[]
        baseline_messages_sha=baseline_boundary_audit['output_messages_sha256']
        masked_messages_sha=zero_cot['turn_boundary_audit']['output_messages_sha256']
        if baseline_messages_sha!=masked_messages_sha:
            violations.append({'kind':'adapter_messages','baseline_sha256':baseline_messages_sha,
                'masked_sha256':masked_messages_sha,
                'baseline_message_count':baseline_boundary_audit.get('output_messages'),
                'masked_message_count':zero_cot['turn_boundary_audit'].get('output_messages')})
        zero_difference=_first_difference(zero_targets,baseline_targets[:len(zero_targets)])
        if len(zero_targets)>len(baseline_targets) or zero_difference is not None:
            violations.append({'kind':'zero_cot_target_prefix','first_difference':zero_difference,
                'baseline_target_tokens':len(baseline_targets),'masked_target_tokens':len(zero_targets),
                'baseline_sha256':digest(baseline_targets),'masked_sha256':digest(zero_targets)})
        actual_difference=_first_difference(actual_targets,zero_targets[:len(actual_targets)])
        if len(actual_targets)>len(zero_targets) or actual_difference is not None:
            violations.append({'kind':'filled_cot_target_prefix','first_difference':actual_difference,
                'zero_cot_target_tokens':len(zero_targets),'actual_target_tokens':len(actual_targets),
                'zero_cot_sha256':digest(zero_targets),'actual_sha256':digest(actual_targets)})
        if require_exact_zero and (actual['input_ids']!=zero_cot['input_ids'] or actual['labels']!=zero_cot['labels']):
            violations.append({'kind':'zero_generation_exact_arrays','input_equal':actual['input_ids']==zero_cot['input_ids'],
                'labels_equal':actual['labels']==zero_cot['labels']})
        rendered_input = self._rendered_input_difference_audit(zero_cot, actual, retained_texts)
        if not rendered_input['verified']:
            violations.append({'kind':'filled_input_difference',
                'expected_reasoning_count':rendered_input['expected_reasoning_count'],
                'located_reasoning_count':rendered_input['located_reasoning_count'],
                'zero_cot_rendered_sha256':rendered_input['zero_cot_rendered_sha256'],
                'stripped_actual_rendered_sha256':rendered_input['stripped_actual_rendered_sha256']})
        audit={'schema':CROSS_ARM_PARITY_SCHEMA,'verified':not violations,'adapter_divergences':violations,
            'baseline_messages_sha256':baseline_messages_sha,'masked_messages_sha256':masked_messages_sha,
            'baseline_input_tokens':len(baseline['input_ids']),'masked_zero_cot_input_tokens':len(zero_cot['input_ids']),
            'actual_masked_input_tokens':len(actual['input_ids']),
            'baseline_target_tokens':len(baseline_targets),'masked_zero_cot_target_tokens':len(zero_targets),
            'actual_masked_target_tokens':len(actual_targets),
            'zero_cot_target_delta_vs_baseline':len(zero_targets)-len(baseline_targets),
            'actual_target_delta_vs_baseline':len(actual_targets)-len(baseline_targets),
            'target_inflation_limit_tokens':0,'target_sequence_policy':'exact-prefix-after-262144-cutoff/v1',
            'input_difference_policy':'official-xhigh-system-and-filled-think-input-only/v1',
            'rendered_input_difference':rendered_input}
        if violations:raise CrossArmParityError(audit)
        return audit

    @staticmethod
    def _attach_tool_audit(result, tool_audit):
        """Bind independent raw/source tool evidence to the final merged row."""
        if not isinstance(tool_audit, dict) or tool_audit.get('verified') is not True:
            raise ValueError('Missing verified raw tool-cardinality evidence')
        for container in (result['turn_boundary_audit'],
                          result['normalization']['turn_boundary']):
            container['raw_tool_cardinality'] = deepcopy(tool_audit)

    def render_generated_trace(self,original_trace,generated_candidates,generation_config,*,boundaries=None):
        result=super().render_generated_trace(original_trace,generated_candidates,generation_config,boundaries=boundaries)
        # A filled gap deliberately prevents the source adapter from joining that
        # fragment into preceding assistant content, because doing so would move
        # its reasoning ahead of earlier visible text.  If native placement then
        # rejects that CoT, re-render from the immutable source with only the
        # retained entries.  This restores the exact zero-CoT source-fragment
        # joins instead of letting the later continuation normalizer introduce a
        # different separator into the visible target.
        ledger=result['cot_provenance'].get('native_gap_omission',{})
        omitted=ledger.get('omitted_generated_gaps',[])
        if omitted:
            omitted_ids={ref.get('gap_id') for ref in omitted}
            input_ids=[entry.get('gap',{}).get('id') if isinstance(entry,dict) else None
                       for entry in generated_candidates]
            if (None in omitted_ids or None in input_ids or len(set(input_ids))!=len(input_ids)
                    or not omitted_ids <= set(input_ids)):
                raise ValueError('Omitted gaps do not identify bound generation entries')
            retained_entries=[entry for entry,gap_id in zip(generated_candidates,input_ids)
                              if gap_id not in omitted_ids]
            rebuilt=native_parent.Qwen38TurnBoundaryMaskedRenderer.render_generated_trace(
                self,original_trace,retained_entries,generation_config,boundaries=boundaries)
            rebuilt_provenance=rebuilt['cot_provenance']
            rebuilt_ledger=rebuilt_provenance.get('native_gap_omission',{})
            expected_retained=[ref for ref in ledger.get('input_generated_gaps',[])
                               if ref.get('gap_id') not in omitted_ids]
            if (rebuilt_ledger.get('omitted_generation_count')!=0
                    or rebuilt_provenance.get('generated_gaps')!=expected_retained):
                raise ValueError('Omission-only canonical rebuild changed retained generations')
            rebuilt_mapping=rebuilt_provenance.get('original_event_message_mapping')
            if not isinstance(rebuilt_mapping,list):
                raise ValueError('Omission-only canonical rebuild lacks original event mapping')
            remapped=[]
            for ref in omitted:
                matches=[(index,event_ids) for index,event_ids in enumerate(rebuilt_mapping)
                         if isinstance(event_ids,list) and ref.get('event_id') in event_ids]
                if len(matches)!=1:
                    raise ValueError('Omitted gap does not map uniquely after canonical rebuild')
                index,event_ids=matches[0]
                remapped.append({**deepcopy(ref),'original_message_index':index,
                    'original_event_ids':deepcopy(event_ids)})
            restored_ledger=deepcopy(ledger)
            restored_ledger['omitted_generated_gaps']=remapped
            rebuilt_provenance['native_gap_omission']=restored_ledger
            for key in ('original_generation_missing_gap_count','unfilled_gap_count'):
                if key in result['cot_provenance']:
                    rebuilt_provenance[key]=result['cot_provenance'][key]
            rebuilt['cot_mask_audit']['bound_input_generation_count']=ledger['bound_input_generation_count']
            rebuilt['cot_mask_audit']['omitted_generation_count']=ledger['omitted_generation_count']
            result=rebuilt
        try:
            zero_cot=native_parent.Qwen38TurnBoundaryMaskedRenderer.render_generated_trace(
                self,original_trace,[],generation_config,boundaries=boundaries)
            source=validate_trace(deepcopy(original_trace))
            baseline_messages,baseline_audit=baseline_source_adapters.canonical_messages(source['events'])
            candidate_texts={entry.get('gap',{}).get('id'):entry.get('candidate',{}).get('text')
                             for entry in generated_candidates if isinstance(entry,dict)}
            retained_texts=[]
            for ref in result['cot_provenance']['generated_gaps']:
                text=candidate_texts.get(ref['gap_id'])
                if not isinstance(text,str) or digest(text)!=ref['candidate_sha256']:
                    raise ValueError('Retained generation text is not bound to its exact candidate receipt')
                retained_texts.append(text)
            parity=self._cross_arm_audit(baseline_messages,baseline_audit['turn_boundary_audit'],zero_cot,result,
                require_exact_zero=not generated_candidates,retained_texts=retained_texts)
        except CrossArmParityError:raise
        except Exception as error:raise _parity_stage_error('trace_no_cot_cross_arm',error) from error
        result['cot_provenance']['cross_arm_no_cot_parity']=parity
        self._attach_tool_audit(
            result, baseline_audit['turn_boundary_audit']['raw_tool_cardinality'])
        return result

    def render_replay_messages(self,messages,*,source_digest,trace_id,boundaries=None,
                               source_tool_audit=None):
        result=super().render_replay_messages(messages,source_digest=source_digest,trace_id=trace_id,boundaries=boundaries)
        try:
            clean=[];metadata=[];counts=Counter()
            for index,message in enumerate(messages):
                row={k:deepcopy(v) for k,v in message.items() if k in native_parent._METADATA_KEYS}
                for key,value in ((boundaries[index] if boundaries is not None else {}) or {}).items():
                    if key in row and row[key]!=value:raise ValueError('Supplied boundary contradicts source metadata')
                    row[key]=deepcopy(value)
                payload={k:deepcopy(v) for k,v in message.items() if k not in native_parent._METADATA_KEYS}
                payload['content']=self._baseline._content(payload.get('content'),payload.get('role'),counts)
                clean.append(payload);metadata.append(row)
            baseline_messages,baseline_audit=normalize.coalesce_assistant_continuations(
                clean,boundaries=metadata,reasoning_policy='drop')
            parity=self._cross_arm_audit(baseline_messages,baseline_audit,result,result,
                require_exact_zero=True,retained_texts=())
        except CrossArmParityError:raise
        except Exception as error:raise _parity_stage_error('replay_no_cot_cross_arm',error) from error
        result['cot_provenance']['cross_arm_no_cot_parity']=parity
        if source_tool_audit is None:
            _, source_audit = source_adapters._finish(
                deepcopy(messages), deepcopy(boundaries) if boundaries is not None
                else [{} for _ in messages], Counter())
            source_tool_audit = source_audit['turn_boundary_audit']['raw_tool_cardinality']
        self._attach_tool_audit(result, source_tool_audit)
        return result

    def _render_coalesced(self,messages,refs,provenance,counts,*,reinsert_generated,boundaries):
        working=deepcopy(messages);clean=[];metadata=[];extra_counts=Counter()
        for i,message in enumerate(working):
            row={k:deepcopy(v) for k,v in message.items() if k in native_parent._METADATA_KEYS}
            extra=boundaries[i] if boundaries is not None else {}
            for k,v in (extra or {}).items():
                if k in row and row[k]!=v:raise ValueError('Supplied boundary contradicts source metadata')
                row[k]=deepcopy(v)
            payload={k:deepcopy(v) for k,v in message.items() if k not in native_parent._METADATA_KEYS}
            payload['content']=self._baseline._content(payload.get('content'),payload.get('role'),extra_counts)
            if not reinsert_generated:payload.pop('reasoning_content',None)
            clean.append(payload);metadata.append(row)
        try:pruned,omissions=prune_conflicting_reasoning(clean,metadata)
        except normalize.ContinuationMappingError as error:
            raise MaskedBoundaryExclusion(error.code,provenance=provenance,filled_gaps=len(refs),audit=error.audit) from error
        original_refs=deepcopy(refs);original_mapping=provenance.get('event_message_mapping',[])
        omitted_refs=[];seen=set()
        for record in omissions:
            index=record['original_message_index'];ids=original_mapping[index]
            matches=[ref for ref in refs if ref['event_id'] in ids and ref['candidate_sha256']==record['candidate_sha256']]
            if len(matches)!=1 or matches[0]['gap_id'] in seen:raise ValueError('Omitted fragment is not a unique bound original gap')
            seen.add(matches[0]['gap_id']);omitted_refs.append({**deepcopy(matches[0]),**record,'original_event_ids':deepcopy(ids)})
            working[index].pop('reasoning_content',None)
        kept=[ref for ref in refs if ref['gap_id'] not in seen]
        safe_provenance={**deepcopy(provenance),'generated_gaps':deepcopy(kept)}
        result=super()._render_coalesced(working,kept,safe_provenance,counts,
            reinsert_generated=reinsert_generated,boundaries=boundaries)
        ledger={'policy':OMISSION_POLICY,'bound_input_generation_count':len(original_refs),
            'retained_generation_count':len(kept),'omitted_generation_count':len(omitted_refs),
            'input_generated_gaps':original_refs,'omitted_generated_gaps':omitted_refs,
            'original_generation_receipts_changed':False,'original_source_changed':False,
            'trace_retained_when_all_cot_omitted':True,'candidate_semantic_quality_assessed':False}
        if 'original_gap_count' in result['cot_provenance']:
            result['cot_provenance']['original_generation_missing_gap_count']=provenance.get('unfilled_gap_count')
            result['cot_provenance']['unfilled_gap_count']=provenance['original_gap_count']-len(kept)
        result['cot_provenance']['native_gap_omission']=ledger
        result['cot_mask_audit']['bound_input_generation_count']=len(original_refs)
        result['cot_mask_audit']['omitted_generation_count']=len(omitted_refs)
        result['identity']=deepcopy(self.identity)
        return result


def training_row(result,*,group_id,source='trace',provenance=None):
    if result.get('identity',{}).get('mask_policy')!=MASK_POLICY:
        raise ValueError('Expected official-native-gap masked result')
    if result.get('cot_provenance',{}).get('source_kind')!=source or result['cot_provenance'].get('reasoning_relocated') is not False:
        raise ValueError('Missing native gap-preserving source provenance')
    parity=result['cot_provenance'].get('cross_arm_no_cot_parity',{})
    if (parity.get('schema')!=CROSS_ARM_PARITY_SCHEMA or parity.get('verified') is not True
            or parity.get('adapter_divergences')!=[] or parity.get('target_inflation_limit_tokens')!=0
            or parity.get('actual_target_delta_vs_baseline',1)>0):
        raise ValueError('Missing strict baseline-visible cross-arm parity')
    row=base_training_row(result,group_id=group_id,source=source,provenance={**(provenance or {}),'generated_cot':result['cot_provenance']})
    for key in ('cot_mask_audit','turn_boundary_audit'):
        row['metadata'][key]=deepcopy(result[key])
    return row
