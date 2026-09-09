"""Official native reasoning placement, omitting only conflicting generated gaps.

Original source/generation receipts are immutable. A omitted insertion is not a
rejection of its trace or a semantic review of the candidate.
"""
from copy import deepcopy
from collections import Counter
import hashlib
from pathlib import Path

from cot_filler.core import digest
from training.qwen38_no_cot.render import training_row as base_training_row
from training.qwen38_turn_boundary_v2 import masked as native_parent
from . import normalize,source_adapters

VERSION='qwen3.8-xhigh-native-leading-gap-cot-input-only/v3'
MASK_POLICY='assistant_content_and_eos_only_native_gap_cot_masked_v3'
OMISSION_POLICY='keep-native-leading-gap-omit-only-conflicting-cot/v3'
MaskedBoundaryExclusion=native_parent.MaskedBoundaryExclusion


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            reasoning_placement_policy=OMISSION_POLICY,omitted_gap_policy=OMISSION_POLICY,
            inline_reasoning_inserted=False,ordinary_native_assistant_template=True)
        self.identity['adaptations']=[x for x in self.identity['adaptations'] if x!='exclude_whole_source_if_filled_reasoning_would_move']+['omit_only_conflicting_generated_gaps_retain_visible_trace']

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
    row=base_training_row(result,group_id=group_id,source=source,provenance={**(provenance or {}),'generated_cot':result['cot_provenance']})
    for key in ('cot_mask_audit','turn_boundary_audit'):
        row['metadata'][key]=deepcopy(result[key])
    return row
