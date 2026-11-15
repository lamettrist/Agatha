from types import SimpleNamespace as NS
from unittest.mock import Mock
import json
from agatha.assistant import Assistant,TOOLS
from agatha.properties import PropertyLookup,PropertyExtraction,verified
from agatha.storage import Store

class ToolMessage:
    def __init__(self,name,args):
        self.tool_calls=[] if name is None else [NS(id='call_'+name,function=NS(name=name,arguments=json.dumps(args)))]
        self.content='Untrusted prose claiming 999 objects fell'
    def model_dump(self,**kwargs):
        return dict(role='assistant',tool_calls=[dict(id=c.id,type='function',function=dict(name=c.function.name,arguments=c.function.arguments)) for c in self.tool_calls])

def test_unavailable_keys_and_exact_tools(tmp_path,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False); monkeypatch.delenv('EXA_API_KEY',raising=False)
    store=Store(tmp_path); key=store.demo()['scene_id']; agent=Assistant(store)
    assert agent.query(key,'push the mug')['status']=='unavailable'
    assert agent.lookup.lookup('cup')['mass_source']=='estimated'
    assert {t['function']['name'] for t in TOOLS}=={'get_scene','lookup_object_properties','simulate','world_model_rollout'}

def test_exa_structured_extraction_and_cache(tmp_path):
    extraction=PropertyExtraction(mass_kg=.35,size_m=None,source_url='https://manufacturer.example/cup',mass_evidence='Net weight: 350 g',size_evidence=None,product_name='Test cup')
    searcher=NS(search=Mock(return_value=NS(results=[NS(url=extraction.source_url,title='Specs',text='Product specs. Net weight: 350 g. Ceramic cup.')])) )
    client=NS(chat=NS(completions=NS(parse=Mock(return_value=NS(choices=[NS(message=NS(parsed=extraction))])))))
    lookup=PropertyLookup(Store(tmp_path)); result=lookup.lookup('cup',client,searcher)
    assert result['mass_source']=='exa' and result['mass_kg']==.35
    assert lookup.lookup('cup')['cached'] is True
    assert searcher.search.call_count==1
    assert PropertyLookup(Store(tmp_path)).lookup('cup')['mass_source']=='exa'
    extraction.mass_kg=3.5
    assert not verified(extraction,[dict(url=extraction.source_url,text='Net weight: 350 g')])
    extraction.mass_kg=.35; extraction.source_url='https://invented.example'
    assert not verified(extraction,[dict(url='https://manufacturer.example/cup',text='Net weight: 350 g')])

def test_agent_runs_real_tools_and_grounds_output(tmp_path,monkeypatch):
    monkeypatch.delenv('EXA_API_KEY',raising=False)
    store=Store(tmp_path);record=store.demo();agent=Assistant(store)
    actions=[dict(type='push',id='mug',direction=[1,0],impulse_Ns=.55)]
    messages=[ToolMessage('get_scene',{}),ToolMessage('lookup_object_properties',{'label':'cup'}),ToolMessage('simulate',{'actions':actions}),ToolMessage('world_model_rollout',{'actions':actions}),ToolMessage(None,{})]
    client=NS(chat=NS(completions=NS(create=Mock(side_effect=[NS(choices=[NS(message=m)]) for m in messages]))))
    result=agent.query(record['scene_id'],'Push the mug',client)
    assert result['status']=='ok'
    assert len(result['actions_ran'])==2 and '999' not in result['answer']
    assert result['simulation_results'][0]['fell']==['mug']
    assert 'estimated mass' in result['answer']
    assert store.get(record['scene_id'])==record
