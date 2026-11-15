import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const $=id=>document.getElementById(id);
let record=null,result=null,busy=false,playing=false,playStart=0,currentIndex=0;
const colors=[0xa9f1cf,0xc6b9ff,0xffc98a,0x9abce5,0xe5a5b5,0xd9dd9a];
const views=[];
const status=message=>$('status').textContent=message;
async function api(path,body){
  const options=body instanceof FormData?{method:'POST',body}:body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
  const response=await fetch(path,options);const data=await response.json();
  if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;
}
function setBusy(value){busy=value;for(const id of ['load-demo','run','photo','save-score'])$(id).disabled=value;}
function safely(fn){return async event=>{try{await fn(event);}catch(error){status(error.message);console.error(error);}};}
function buildView(engine){
  const container=$(engine+'-view'),scene=new THREE.Scene();scene.background=new THREE.Color('#111923');
  const camera=new THREE.PerspectiveCamera(43,1,.01,40);camera.position.set(1.55,1.6,1.85);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;container.append(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xdaeeff,0x233044,2.5));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(2,4,2);light.castShadow=true;light.shadow.mapSize.set(1024,1024);scene.add(light);
  const grid=new THREE.GridHelper(5,30,0x344552,0x202e3b);grid.position.y=.001;scene.add(grid);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,.63,0);controls.enableDamping=true;controls.minDistance=.5;controls.maxDistance=6;controls.maxPolarAngle=Math.PI*.49;
  const group=new THREE.Group();scene.add(group);const meshes=new Map();let support=null;
  const view={engine,renderer,scene,camera,controls,group,meshes,container,get support(){return support;},set support(v){support=v;}};
  const resize=()=>{const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();};
  new ResizeObserver(resize).observe(container);resize();
  let down=null;
  renderer.domElement.addEventListener('pointerdown',event=>{down=[event.clientX,event.clientY];});
  renderer.domElement.addEventListener('pointerup',safely(async event=>{
    if(busy||!record||!down||Math.hypot(event.clientX-down[0],event.clientY-down[1])>5)return;
    const rect=renderer.domElement.getBoundingClientRect();const mouse=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
    const ray=new THREE.Raycaster();ray.setFromCamera(mouse,camera);const hit=ray.intersectObjects([...meshes.values()].filter(mesh=>mesh.visible))[0];if(!hit)return;
    const picked=hit.object.userData.id?hit.object:hit.object.parent;const id=picked?.userData.id;const object=record.scene.objects.find(o=>o.id===id);if(!object||object.is_static)return;
    $('object').value=id;updateMeta();await run([{type:'push',id,direction:[1,0],impulse_Ns:Number($('action').value==='push'?$('amount').value:.55)}]);
  }));
  return view;
}
function box(size,color){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),new THREE.MeshStandardMaterial({color,roughness:.68,metalness:.06}));mesh.castShadow=true;mesh.receiveShadow=true;const edge=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.15}));mesh.add(edge);return mesh;}
function rebuild(){
  for(const view of views){
    view.group.traverse(object=>{object.geometry?.dispose();if(object.material){for(const m of Array.isArray(object.material)?object.material:[object.material])m.dispose();}});
    view.group.clear();view.meshes.clear();
    const s=record.scene.support;view.support=box([s.size[0],.08,s.size[1]],0x4d5c69);view.support.position.set(0,s.height-.04,0);view.group.add(view.support);
    // The demo desk object represents the support. Rendering it twice would z-fight.
    record.scene.objects.forEach((o,i)=>{if(o.id==='desk'&&o.is_static)return;const mesh=box(o.size,colors[i%colors.length]);mesh.position.fromArray(o.position);mesh.rotation.set(...o.rotation_euler,'ZYX');mesh.userData.id=o.id;view.group.add(mesh);view.meshes.set(o.id,mesh);});
    const scale=Math.max(...s.size)/1.2;view.camera.position.set(1.55*scale,s.height+.85*scale,1.85*scale);view.controls.target.set(0,s.height-.12,0);
  }
}
function updateMeta(){const o=record?.scene.objects.find(o=>o.id===$('object').value);$('object-meta').textContent=o?`${o.label} · ${o.mass_kg.toFixed(2)} kg (${o.mass_source}) · ${o.size.map(v=>v.toFixed(2)).join(' × ')} m`:'';}
function loadRecord(data){record=data;result=null;playing=false;currentIndex=0;$('timeline').value=0;$('timeline').max=90;$('time').textContent='0.00 / 3.00 s';$('object').replaceChildren();for(const o of record.scene.objects.filter(o=>!o.is_static)){const option=new Option(`${o.label} / ${o.id}`,o.id);$('object').add(option);}if(record.scene.objects.some(o=>o.id==='mug'))$('object').value='mug';updateMeta();rebuild();for(const id of ['fell','chaos','error'])$(id).textContent='—';$('scene-label').textContent=`${record.report?.source==='bundled_demo'?'OFFLINE DESK':'PHOTO SCENE'} / ${record.scene.objects.length} BOXES`;$('report').textContent=JSON.stringify(record.report,null,2);status('Ready. Click an object to push it, or choose an intervention.');}
async function loadDemo(){if(busy)return;setBusy(true);try{loadRecord(await api('/api/scenes/demo',{}));$('photo-wrap').hidden=true;$('reconstruction').textContent='Bundled desk · all masses and dimensions are estimated. No network needed.';}finally{setBusy(false);}}
function actions(){const type=$('action').value,id=$('object').value,amount=Number($('amount').value);if(type==='push')return[{type,id,direction:[1,0],impulse_Ns:amount}];if(type==='remove')return[{type,id}];if(type==='set_mass')return[{type,id,kg:amount}];if(type==='tilt_support')return[{type,angle_deg:amount,axis:'z'}];return[{type,g:amount}];}
async function run(interventions=actions()){
  if(!record||busy)return;setBusy(true);playing=false;status('Running PyBullet + learned autoregressive rollout…');
  try{result=await api('/api/simulate',{scene_id:record.scene_id,actions:interventions,engine:'both',duration:3,fps:30});$('fell').textContent=result.physics.events.fell.length;$('chaos').textContent=result.physics.chaos_score.toFixed(3);status(`Physics fell: ${result.physics.events.fell.join(', ')||'none'}. Learned fell: ${result.learned?.events.fell.join(', ')||'none'}. ${(result.warnings||[]).join(' ')}`);$('timeline').max=result.physics.frames.length-1;startPlayback();}finally{setBusy(false);}
}
function applyFrame(index){
  currentIndex=index;if(!result)return;const physics=result.physics?.frames[index];if(!physics)return;$('timeline').value=index;$('time').textContent=`${physics.t.toFixed(2)} / ${result.physics.frames.at(-1).t.toFixed(2)} s`;
  for(const view of views){const prediction=result[view.engine];const f=prediction?.frames[index];const states=new Map((f?.objects||[]).map(o=>[o.id,o]));for(const [id,mesh] of view.meshes){const s=states.get(id);mesh.visible=!!s;if(s){mesh.position.fromArray(s.position);mesh.quaternion.fromArray(s.quaternion);}}if(prediction)view.support.rotation.set(...prediction.support_rotation_euler,'ZYX');}
  $('error').textContent=result.error?`${result.error.per_frame[index].mean_position_error_m.toFixed(3)} m`:'unavailable';
}
function startPlayback(){if(!result)return;playing=true;playStart=performance.now();$('play').textContent='Ⅱ';applyFrame(0);}
function animate(now){requestAnimationFrame(animate);if(playing&&result){const index=Math.min(Math.floor((now-playStart)/1000*30),result.physics.frames.length-1);applyFrame(index);if(index===result.physics.frames.length-1){playing=false;$('play').textContent='▶';}}for(const view of views){if(view.container.clientWidth&&view.container.clientHeight){view.controls.update();view.renderer.render(view.scene,view.camera);}}}
async function upload(){const file=$('photo').files[0];if(!file||busy)return;setBusy(true);status('Detecting objects, estimating relative depth, fitting the support plane…');try{const form=new FormData();form.append('image',file);if($('table-width').value)form.append('table_width_m',$('table-width').value);const data=await api('/api/scenes',form);loadRecord(data);const report=data.report;$('reconstruction').textContent=`${report.detected_count} detected · ${report.kept_count} kept · scale confidence ${report.scale_confidence} · ${report.depth}. ${report.warnings.join(' ')}`;const image=new Image();const url=URL.createObjectURL(file);image.src=url;await image.decode();const canvas=$('detections');[canvas.width,canvas.height]=report.image_size;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0,canvas.width,canvas.height);ctx.font=`${Math.max(12,canvas.width/45)}px sans-serif`;ctx.lineWidth=Math.max(2,canvas.width/400);for(const d of report.detections){const [x,y,x2,y2]=d.bbox;ctx.strokeStyle=d.object_id?'#74ffc4':'#d0d5db';ctx.fillStyle=ctx.strokeStyle;ctx.strokeRect(x,y,x2-x,y2-y);ctx.fillText(`${d.label} ${Math.round(d.confidence*100)}%`,x,Math.max(16,y-5));}$('photo-wrap').hidden=false;URL.revokeObjectURL(url);if(!report.kept_count)status('No supported objects reconstructed. Check the report, try another photo, or load the offline desk.');}finally{setBusy(false);}}
async function scores(){const data=await api('/api/leaderboard');$('scores').replaceChildren();for(const entry of data.entries){const li=document.createElement('li'),score=document.createElement('span');li.textContent=entry.name;score.textContent=entry.chaos_score.toFixed(3);li.append(score);$('scores').append(li);}}
$('load-demo').addEventListener('click',safely(loadDemo));$('run').addEventListener('click',safely(()=>run()));$('photo').addEventListener('change',safely(upload));$('object').addEventListener('change',updateMeta);
$('action').addEventListener('change',()=>{const type=$('action').value;$('amount-field').hidden=type==='remove';$('amount').value={push:.55,set_mass:.8,tilt_support:15,set_gravity:3}[type]??.55;$('amount').min=type==='tilt_support'?-60:type==='set_gravity'?0:.01;$('amount').max={push:20,set_mass:1000,tilt_support:60,set_gravity:30}[type]??20;});
$('engine').addEventListener('change',()=>{const engine=$('engine').value;for(const view of views)$(view.engine+'-panel').hidden=engine!=='both'&&engine!==view.engine;$('views').style.gridTemplateColumns=engine==='both'?'1fr 1fr':'1fr';});
$('timeline').addEventListener('input',()=>{playing=false;$('play').textContent='▶';applyFrame(Number($('timeline').value));});$('play').addEventListener('click',()=>{if(playing){playing=false;$('play').textContent='▶';}else if(result&&currentIndex<result.physics.frames.length-1&&currentIndex>0){playing=true;playStart=performance.now()-currentIndex/30*1000;$('play').textContent='Ⅱ';}else startPlayback();});
$('ask-form').addEventListener('submit',safely(async event=>{event.preventDefault();if(!record)return;$('ask').disabled=true;$('answer').textContent='Agent is reading the scene, looking up properties, and running tools…';$('sources').replaceChildren();try{const data=await api('/api/assistant/query',{scene_id:record.scene_id,question:$('question').value});$('answer').textContent=data.answer+(data.warnings?.length?' '+data.warnings.join(' '):'');for(const source of data.sources){if(!/^https?:\/\//.test(source.url))continue;const a=document.createElement('a');a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=`↗ ${source.label} · ${source.mass_kg} kg · ${source.url}`;$('sources').append(a);}$('trace-wrap').hidden=false;$('trace').textContent=JSON.stringify({actions_ran:data.actions_ran,tool_trace:data.tool_trace},null,2);}catch(error){$('answer').textContent=error.message;}finally{$('ask').disabled=false;}}));
$('score-form').addEventListener('submit',safely(async event=>{event.preventDefault();if(!record||!result?.physics){status('Run physics before submitting a score.');return;}await api('/api/leaderboard',{name:$('name').value,scene_id:record.scene_id,chaos_score:result.physics.chaos_score});await scores();status('Physics score submitted.');}));
try{views.push(buildView('physics'),buildView('learned'));requestAnimationFrame(animate);await loadDemo();await scores();}catch(error){status(`Demo initialization failed: ${error.message}. WebGL is required for the 3D view; API docs at /docs.`);console.error(error);}
