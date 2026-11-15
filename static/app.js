import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {GLTFLoader} from './vendor/GLTFLoader.js';
const gltfLoader=new GLTFLoader();

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
function setBusy(value){busy=value;for(const id of ['load-demo','load-bottle','load-laptop','load-soft','run','photo','save-score'])$(id).disabled=value;}
function safely(fn){return async event=>{try{await fn(event);}catch(error){status(error.message);console.error(error);}};}
function buildView(engine){
  const container=$(engine+'-view'),scene=new THREE.Scene();scene.background=new THREE.Color('#111923');
  const camera=new THREE.PerspectiveCamera(43,1,.01,40);camera.position.set(1.55,1.6,1.85);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;container.append(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xdaeeff,0x233044,2.5));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(2,4,2);light.castShadow=true;light.shadow.mapSize.set(1024,1024);scene.add(light);
  const grid=new THREE.GridHelper(5,30,0x344552,0x202e3b);grid.position.y=.001;scene.add(grid);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,.63,0);controls.enableDamping=true;controls.minDistance=.5;controls.maxDistance=6;controls.maxPolarAngle=Math.PI*.49;
  const group=new THREE.Group();scene.add(group);const meshes=new Map();let support=null;
  const view={engine,renderer,scene,camera,controls,group,meshes,soft:new Map(),joints:[],container,get support(){return support;},set support(v){support=v;}};
  const resize=()=>{const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();};
  new ResizeObserver(resize).observe(container);resize();
  let down=null;
  renderer.domElement.addEventListener('pointerdown',event=>{down=[event.clientX,event.clientY];});
  renderer.domElement.addEventListener('pointerup',safely(async event=>{
    if(busy||!record||!down||Math.hypot(event.clientX-down[0],event.clientY-down[1])>5)return;
    const rect=renderer.domElement.getBoundingClientRect();const mouse=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
    const ray=new THREE.Raycaster();ray.params.Line.threshold=.004;ray.setFromCamera(mouse,camera);const hit=ray.intersectObjects([...meshes.values(),...softPickables(view)].filter(mesh=>mesh.visible),true)[0];if(!hit)return;
    let picked=hit.object;while(picked&&!picked.userData.id)picked=picked.parent;const id=picked?.userData.id;const object=record.scene.objects.find(o=>o.id===id)||softBodies().find(o=>o.id===id);if(!object||object.is_static)return;
    $('object').value=id;updateMeta();if($('click-mode').value==='push')await run([{type:'push',id,direction:[1,0],impulse_Ns:Number($('action').value==='push'?$('amount').value:.55)}]);
  }));
  return view;
}
function box(size,color){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),new THREE.MeshStandardMaterial({color,roughness:.68,metalness:.06}));mesh.castShadow=true;mesh.receiveShadow=true;const edge=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.15}));mesh.add(edge);return mesh;}
function bodyVisual(o,color){
  const group=new THREE.Group();
  const parts=o.collision?.parts||[{type:o.shape==='cylinder'?'cylinder':'box',size:o.size,radius:o.size[0]/2,height:o.size[1]}];
  for(const part of parts){
    let mesh;
    if(part.type==='cylinder')mesh=new THREE.Mesh(new THREE.CylinderGeometry(part.radius,part.radius,part.height,40),new THREE.MeshStandardMaterial({color,roughness:.7}));
    else if(part.type==='convex_hull')mesh=box(o.size,color);
    else mesh=box(part.size,color);
    mesh.position.fromArray(part.position||[0,0,0]);mesh.rotation.set(...(part.rotation_euler||[0,0,0]),'ZYX');mesh.castShadow=true;mesh.receiveShadow=true;group.add(mesh);
  }
  if(o.visual_mesh_url)gltfLoader.load(o.visual_mesh_url,gltf=>{
    // A late load must not attach assets to a replacement scene.
    if(!group.parent)return;
    group.traverse(child=>{child.geometry?.dispose();child.material?.dispose?.();});group.clear();
    gltf.scene.traverse(child=>{if(child.isMesh){child.castShadow=true;child.receiveShadow=true;if(o.confidence<.5&&child.material?.color)child.material.color.set('#ffaf55');}});group.add(gltf.scene);
  },undefined,error=>console.warn('Visual asset unavailable',o.id,error));
  return group;
}
function rebuild(){
  for(const view of views){
    view.group.traverse(object=>{object.geometry?.dispose();if(object.material){for(const m of Array.isArray(object.material)?object.material:[object.material])m.dispose();}});
    view.group.clear();view.meshes.clear();view.soft.clear();view.joints=[];
    const s=record.scene.support;view.support=box([s.size[0],s.thickness||.08,s.size[1]],0x4d5c69);view.support.position.set(0,s.height-(s.thickness||.08)/2,0);view.group.add(view.support);
    // The demo desk object represents the support. Rendering it twice would z-fight.
    record.scene.objects.forEach((o,i)=>{if(o.id==='desk'&&o.is_static)return;const mesh=bodyVisual(o,o.confidence<.5?0xffaf55:colors[i%colors.length]);mesh.position.fromArray(o.position);mesh.rotation.set(...o.rotation_euler,'ZYX');mesh.userData.id=o.id;view.group.add(mesh);view.meshes.set(o.id,mesh);});
    buildRouted(view);
    const scale=Math.max(...s.size)/1.2;view.camera.position.set(1.55*scale,s.height+.85*scale,1.85*scale);view.controls.target.set(0,s.height-.12,0);
  }
}
function updateMeta(){const soft=softBodies().find(o=>o.id===$('object').value);if(soft){$('object-meta').textContent=`${soft.label} · soft ${soft.kind} · ${soft.mass_kg.toFixed(2)} kg · ${soft.size.map(v=>v.toFixed(2)).join(' × ')} m · stiffness ${soft.stiffness} · learned view only unless a reference exists`;$('edit-mass').value=soft.mass_kg;$('edit-shape').value='keep';return;}const o=record?.scene.objects.find(o=>o.id===$('object').value);$('object-meta').textContent=o?`${o.label} · ${o.mass_kg.toFixed(2)} kg (${o.mass_source}) · ${o.size.map(v=>v.toFixed(2)).join(' × ')} m · ${o.shape_method||'box'}${o.soft_approx?' · treated as rigid':''}`:'';if(o)$('edit-mass').value=o.mass_kg;$('edit-shape').value='keep';}

function loadRecord(data){record=data;physicsLabel(true);$('action').value='push';$('amount').value='.55';$('amount-field').hidden=false;$('table-collisions').checked=record.scene.support.collisions_enabled!==false;result=null;playing=false;currentIndex=0;$('timeline').value=0;$('timeline').max=90;$('time').textContent='0.00 / 3.00 s';$('object').replaceChildren();for(const o of record.scene.objects.filter(o=>!o.is_static)){const option=new Option(`${o.label} / ${o.id}`,o.id);$('object').add(option);}for(const o of softBodies())$('object').add(new Option(`${o.label} / ${o.id} (soft)`,o.id));{const routed=isRouted(),select=$('architecture');if(routed)select.value='auto';else if(select.value==='auto')select.value='default';if(routed)$('model-label').textContent='Auto (router) · run to route';else if(/experimental|router/.test($('model-label').textContent))$('model-label').textContent='Learned dynamics';}if(record.scene.objects.some(o=>o.id==='mug'))$('object').value='mug';updateMeta();rebuild();for(const id of ['fell','chaos','error'])$(id).textContent='—';$('scene-label').textContent=`${record.report?.source==='bundled_demo'?(({bottle:'BOTTLE EXPERIMENT',laptop:'LAPTOP / JOINTS',soft:'SOFT BODY'})[record.report?.scenario]||'OFFLINE DESK'):'PHOTO SCENE'} / ${record.scene.objects.length} BODIES`;$('report').textContent=JSON.stringify(record.report,null,2);$('scene-view').value='scene';$('photo-stage').hidden=true;$('views').hidden=false;status('Ready. Select an object to inspect it, or choose an intervention.');}
async function loadDemo(scenario='desk'){if(busy)return;setBusy(true);try{loadRecord(await api('/api/scenes/demo?scenario='+scenario,{}));$('photo-wrap').hidden=true;$('reconstruction').textContent='Bundled '+scenario+' · all masses and dimensions are estimated. No network needed.';if(scenario==='bottle'){$('question').value='What if I toss the bottle upward with table collisions disabled? Look up its mass and compare both engines.';$('action').value='toss';$('amount').value='1.5';$('amount-field').hidden=false;}else{$('action').value='push';$('amount').value='.55';$('amount-field').hidden=false;}}finally{setBusy(false);}}
function actions(){const type=$('action').value,id=$('object').value,amount=Number($('amount').value);if(type==='toss')return[{type:'push',id,direction:[0,1,0],impulse_Ns:amount}];if(type==='set_support_collisions'||type==='enable_support_collisions')return[{type:'set_support_collisions',enabled:type==='enable_support_collisions'}];if(type==='push')return[{type,id,direction:[1,0],impulse_Ns:amount}];if(type==='remove')return[{type,id}];if(type==='set_mass')return[{type,id,kg:amount}];if(type==='tilt_support')return[{type,angle_deg:amount,axis:'z'}];return[{type,g:amount}];}
async function run(interventions=actions()){
  if(!record||busy)return;if(!$('table-collisions').checked&&!interventions.some(a=>a.type==='set_support_collisions'))interventions=[{type:'set_support_collisions',enabled:false},...interventions];setBusy(true);playing=false;status('Running PyBullet + learned autoregressive rollout…');
  try{result=await api('/api/simulate',{scene_id:record.scene_id,actions:interventions,engine:'both',learned_architecture:$('architecture').value,duration:3,fps:30});$('model-label').textContent=modelLabel(result.learned);physicsLabel(!!result.physics);const ph=result.physics;$('fell').textContent=ph?ph.events.fell.length:'—';$('chaos').textContent=ph?ph.chaos_score.toFixed(3):'—';status(`${(ph||result.learned)?.support_collisions_enabled===false?'Table collisions disabled. ':''}${ph?`Physics fell: ${ph.events.fell.join(', ')||'none'}.`:'PyBullet reference unavailable for this scene type.'} Learned fell: ${result.learned?.events?.fell.join(', ')||'none'}. ${(result.warnings||[]).join(' ')}`);$('timeline').max=Math.max(timelineFrames().length-1,0);startPlayback();}finally{setBusy(false);}
}
function applyFrame(index){
  currentIndex=index;if(!result)return;const timeline=timelineFrames(),physics=timeline[index];if(!physics)return;$('timeline').value=index;$('time').textContent=`${physics.t.toFixed(2)} / ${timeline.at(-1).t.toFixed(2)} s`;
  for(const view of views){const prediction=result[view.engine];if(!prediction){updateJoints(view);continue;}const f=prediction?.frames[index];const states=new Map((f?.objects||[]).map(o=>[o.id,o]));for(const [id,mesh] of view.meshes){const s=states.get(id);mesh.visible=!!s;if(s){mesh.position.fromArray(s.position);mesh.quaternion.fromArray(s.quaternion);}}if(prediction){view.support.rotation.set(...prediction.support_rotation_euler,'ZYX');view.support.material.transparent=prediction.support_collisions_enabled===false;view.support.material.opacity=prediction.support_collisions_enabled===false?.18:1;view.support.material.depthWrite=prediction.support_collisions_enabled!==false;}applySoft(view,prediction,f);updateJoints(view);}
  const frameError=result.error?.per_frame?.[index]?.mean_position_error_m;$('error').textContent=typeof frameError==='number'?`${frameError.toFixed(3)} m`:'unavailable';
}
function startPlayback(){if(!result)return;playing=true;playStart=performance.now();$('play').textContent='Ⅱ';applyFrame(0);}
function animate(now){requestAnimationFrame(animate);if(playing&&result){const last=timelineFrames().length-1,index=Math.min(Math.floor((now-playStart)/1000*30),last);applyFrame(index);if(index>=last){playing=false;$('play').textContent='▶';}}for(const view of views){if(view.container.clientWidth&&view.container.clientHeight){view.controls.update();view.renderer.render(view.scene,view.camera);}}}
async function upload(){const file=$('photo').files[0];if(!file||busy)return;setBusy(true);status('Finding each object, measuring the scene and checking that bodies rest stably…');try{const form=new FormData();form.append('image',file);if($('table-width').value)form.append('table_width_m',$('table-width').value);const extension=file.name.split('.').at(-1).toLowerCase();const route=file.type.startsWith('video/')?'/api/scenes/video':['ply','obj','usdz'].includes(extension)?'/api/scenes/scan':'/api/scenes';if(route!=='/api/scenes'){form.delete('image');form.append('file',file);}const data=await api(route,form);loadRecord(data);const report=data.report;$('reconstruction').textContent=`${report.detected_count} detected · ${report.kept_count} kept · scale confidence ${report.scale_confidence} · ${report.depth}. ${(report.warnings||[]).join(' ')}`;if(!file.type.startsWith('image/')){$('photo-wrap').hidden=true;return;}const image=new Image();const url=URL.createObjectURL(file);image.src=url;await image.decode();const canvas=$('detections');[canvas.width,canvas.height]=report.image_size;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0,canvas.width,canvas.height);ctx.font=`${Math.max(12,canvas.width/45)}px sans-serif`;ctx.lineWidth=Math.max(2,canvas.width/400);for(const d of report.detections||[]){if(!d.bbox)continue;const [x,y,x2,y2]=d.bbox;ctx.strokeStyle=d.object_id?'#74ffc4':'#d0d5db';ctx.fillStyle=ctx.strokeStyle;ctx.strokeRect(x,y,x2-x,y2-y);ctx.fillText(`${d.label} ${Math.round(d.confidence*100)}%`,x,Math.max(16,y-5));}$('photo-wrap').hidden=false;URL.revokeObjectURL(url);if(!report.kept_count)status('No supported objects reconstructed. Check the report, try another photo, or load the offline desk.');}finally{setBusy(false);}}
async function scores(){const data=await api('/api/leaderboard');$('scores').replaceChildren();for(const entry of data.entries){const li=document.createElement('li'),score=document.createElement('span');li.textContent=entry.name;score.textContent=entry.chaos_score.toFixed(3);li.append(score);$('scores').append(li);}}
$('scene-view').addEventListener('change',()=>{const mode=$('scene-view').value;const debug=record?.report?.debug_url;if(mode!=='scene'&&!debug){status('Original image is available after reconstructing a photo or video.');$('scene-view').value='scene';return;}$('photo-stage').hidden=mode==='scene';$('views').hidden=mode!=='scene';if(mode!=='scene')$('stage-image').src=debug+(mode==='photo'?'/original.jpg':'/reprojection.png');});
$('save-object').addEventListener('click',safely(async()=>{if(!record||busy)return;if(softBodies().some(o=>o.id===$('object').value)){status('Soft bodies cannot be edited here.');return;}if((record.scene.joints||[]).length){status('Object edits are disabled for jointed scenes: the edit re-settles the scene without its joints.');return;}setBusy(true);try{const body={mass_kg:Number($('edit-mass').value)};if($('edit-shape').value!=='keep')body.shape=$('edit-shape').value;loadRecord(await api('/api/scenes/'+record.scene_id+'/objects/'+encodeURIComponent($('object').value),body));status('Object updated. Contacts and stability checked again.');}finally{setBusy(false);}}));
$('load-demo').addEventListener('click',safely(()=>loadDemo()));$('load-bottle').addEventListener('click',safely(()=>loadDemo('bottle')));$('load-laptop').addEventListener('click',safely(()=>loadDemo('laptop')));$('load-soft').addEventListener('click',safely(()=>loadDemo('soft')));$('run').addEventListener('click',safely(()=>run()));$('photo').addEventListener('change',safely(upload));$('object').addEventListener('change',updateMeta);
$('action').addEventListener('change',()=>{const type=$('action').value;$('amount-field').hidden=['remove','set_support_collisions','enable_support_collisions'].includes(type);$('amount').value={push:.55,toss:1.5,set_mass:.8,tilt_support:15,set_gravity:3}[type]??.55;$('amount').min=type==='tilt_support'?-60:type==='set_gravity'?0:.01;$('amount').max={push:20,toss:20,set_mass:1000,tilt_support:60,set_gravity:30}[type]??20;});
$('engine').addEventListener('change',()=>{const engine=$('engine').value;for(const view of views)$(view.engine+'-panel').hidden=engine!=='both'&&engine!==view.engine;$('views').style.gridTemplateColumns=engine==='both'?'1fr 1fr':'1fr';});
$('timeline').addEventListener('input',()=>{playing=false;$('play').textContent='▶';applyFrame(Number($('timeline').value));});$('play').addEventListener('click',()=>{if(playing){playing=false;$('play').textContent='▶';}else if(result&&currentIndex<timelineFrames().length-1&&currentIndex>0){playing=true;playStart=performance.now()-currentIndex/30*1000;$('play').textContent='Ⅱ';}else startPlayback();});
$('ask-form').addEventListener('submit',safely(async event=>{event.preventDefault();if(!record)return;if(isRouted()){$('answer').textContent='The what-if agent is not connected to the model router yet, so it cannot simulate joints or soft bodies. Use the Run controls for this scene.';return;}$('ask').disabled=true;$('answer').textContent='Agent is reading the scene, looking up properties, and running tools…';$('sources').replaceChildren();try{const data=await api('/api/assistant/query',{scene_id:record.scene_id,question:$('question').value});$('answer').textContent=data.answer+(data.warnings?.length?' '+data.warnings.join(' '):'');for(const source of data.sources){if(!/^https?:\/\//.test(source.url))continue;const a=document.createElement('a');a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=`↗ ${source.label} · ${source.mass_kg} kg · ${source.url}`;$('sources').append(a);}$('trace-wrap').hidden=false;$('trace').textContent=JSON.stringify({actions_ran:data.actions_ran,tool_trace:data.tool_trace},null,2);}catch(error){$('answer').textContent=error.message;}finally{$('ask').disabled=false;}}));
$('score-form').addEventListener('submit',safely(async event=>{event.preventDefault();if(!record||!result?.physics){status('Run physics before submitting a score.');return;}await api('/api/leaderboard',{name:$('name').value,scene_id:record.scene_id,chaos_score:result.physics.chaos_score});await scores();status('Physics score submitted.');}));
// --- Routed scenes: soft bodies and joints (additive; plain box scenes never touch these paths beyond empty loops) ---
function softBodies(){return record?.scene?.soft_bodies||[];}
function physicsLabel(available){$('physics-label').textContent=available?'PyBullet · 240 Hz':'No PyBullet reference · initial pose only';$('physics-view').style.opacity=available?1:.45;}
function isRouted(){return softBodies().length>0||(record?.scene?.joints||[]).length>0;}
// The timeline follows the PyBullet reference when there is one and the learned rollout otherwise.
function timelineFrames(){return result?.physics?.frames||result?.learned?.frames||[];}
function softPickables(view){const list=[];for(const entry of view.soft.values()){list.push(entry.placeholder);if(entry.mesh)list.push(entry.mesh);}return list;}
const SOFT_COLOR=0xff7ab8,JOINT_COLOR=0xffe45c;
function buildRouted(view){
  for(const o of softBodies()){
    const placeholder=new THREE.Mesh(new THREE.BoxGeometry(...o.size.map(v=>Math.max(v,.004))),new THREE.MeshStandardMaterial({color:SOFT_COLOR,transparent:true,opacity:.35,roughness:.8,depthWrite:false}));
    placeholder.add(new THREE.LineSegments(new THREE.EdgesGeometry(placeholder.geometry),new THREE.LineBasicMaterial({color:SOFT_COLOR})));
    placeholder.position.fromArray(o.position);placeholder.rotation.set(...(o.rotation_euler||[0,0,0]),'ZYX');placeholder.userData.id=o.id;view.group.add(placeholder);view.soft.set(o.id,{placeholder,mesh:null});
  }
  for(const joint of record.scene.joints||[]){
    const marker=new THREE.Mesh(new THREE.SphereGeometry(.012,12,8),new THREE.MeshBasicMaterial({color:JOINT_COLOR,depthTest:false,transparent:true,opacity:.95}));marker.renderOrder=5;
    if(joint.type==='hinge'||joint.type==='prismatic'){const axis=new THREE.Vector3(...(joint.axis||[0,0,1])).normalize().multiplyScalar(.07);const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([axis.clone().negate(),axis]),new THREE.LineBasicMaterial({color:JOINT_COLOR,depthTest:false,transparent:true}));line.renderOrder=5;marker.add(line);}
    view.group.add(marker);view.joints.push({joint,marker});
  }
  updateJoints(view);
}
// World anchor = parent pose applied to anchor_parent (a null parent means anchor_parent is already in world space).
function jointAnchor(joint,poseOf){
  if(joint.parent==null)return{position:new THREE.Vector3(...joint.anchor_parent),quaternion:new THREE.Quaternion()};
  const parent=poseOf(joint.parent);if(parent)return{position:new THREE.Vector3(...joint.anchor_parent).applyQuaternion(parent.quaternion).add(parent.position),quaternion:parent.quaternion.clone()};
  const child=poseOf(joint.child);if(child)return{position:new THREE.Vector3(...joint.anchor_child).applyQuaternion(child.quaternion).add(child.position),quaternion:child.quaternion.clone()};
  return null;
}
function updateJoints(view){
  const poseOf=id=>{const mesh=view.meshes.get(id);return mesh&&mesh.visible?mesh:null;};
  for(const {joint,marker} of view.joints){const anchor=jointAnchor(joint,poseOf);marker.visible=!!anchor&&(view.meshes.get(joint.child)?.visible!==false);if(anchor){marker.position.copy(anchor.position);marker.quaternion.copy(anchor.quaternion);}}
}
function softMesh(view,id,faces,count){
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(count*3),3));geometry.setIndex(faces);
  const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:SOFT_COLOR,roughness:.75,side:THREE.DoubleSide,flatShading:true}));mesh.castShadow=true;mesh.frustumCulled=false;
  const wire=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:0xffffff,wireframe:true,transparent:true,opacity:.22}));wire.frustumCulled=false;wire.raycast=()=>{};mesh.add(wire);
  mesh.userData.id=id;view.group.add(mesh);return mesh;
}
// Soft bodies stay as the translucent placeholder until a result for this view carries their vertices.
function applySoft(view,prediction,frame){
  if(!view.soft.size)return;const meshes=prediction?.soft_meshes,states=new Map((frame?.soft_bodies||[]).map(o=>[o.id,o]));
  for(const [id,entry] of view.soft){
    if(!meshes){entry.placeholder.visible=true;if(entry.mesh)entry.mesh.visible=false;continue;}
    const state=states.get(id),vertices=state?.vertices;entry.placeholder.visible=false;
    if(!vertices||!meshes[id]){if(entry.mesh)entry.mesh.visible=false;continue;}
    if(entry.mesh&&entry.mesh.geometry.attributes.position.array.length!==vertices.length){view.group.remove(entry.mesh);entry.mesh.geometry.dispose();entry.mesh.material.dispose();entry.mesh.children[0]?.material.dispose();entry.mesh=null;}
    if(!entry.mesh)entry.mesh=softMesh(view,id,meshes[id].faces||[],vertices.length/3);
    const position=entry.mesh.geometry.attributes.position;position.array.set(vertices);position.needsUpdate=true;entry.mesh.geometry.computeBoundingSphere();entry.mesh.visible=true;
  }
}
window.__agathaDebug={views,get result(){return result;},get record(){return record;}};
// --- World model before/after card (reads /api/metrics; purely additive, never blocks the demo) ---
function modelLabel(learned){
  if(!learned)return'Unavailable';
  {const names={contact_gnn:'Contact GNN',joint_gnn:'Joint GNN',soft_gnn:'Soft GNN',mlp_multistep:'MLP',graph_impulse:'Graph impulses'},arch=String(learned.model_architecture||''),parts=arch.split('+');if(parts.every(p=>names[p])&&(parts.length>1||arch==='joint_gnn'||arch==='soft_gnn'))return parts.map(p=>names[p]).join(' + ')+' · experimental'+(parts.includes('soft_gnn')&&parts.length>1?' (one-way)':'');}
  if(learned.model_architecture==='graph_impulse')return'Graph · 3 message rounds';
  if(learned.model_architecture==='contact_gnn')return'Contact GNN · learned'+(learned.projection_applied===true?' + projection':learned.projection_applied===false?' only (no projection)':'');
  return'MLP · learned dynamics';
}
const DASH='—';
function sig(value,scale=1){if(typeof value!=='number'||!Number.isFinite(value))return DASH;const v=value*scale;if(v===0)return'0';const text=v.toPrecision(3);return text.includes('e')?String(Number(text)):text;}
const pick=(object,...path)=>path.reduce((o,key)=>o==null?undefined:o[key],object);
function el(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function metricsTable(columns,rows){
  // rows: [label, unit, [values], scale, higherIsBetter]
  const table=el('table'),head=el('tr');head.append(el('th','HELD-OUT METRIC'));for(const c of columns)head.append(el('th',c));table.append(el('thead'));table.tHead.append(head);const body=el('tbody');table.append(body);
  for(const [label,unit,values,scale=1,higher=false] of rows){
    const tr=el('tr'),name=el('td',label+' ');name.append(el('small',unit));tr.append(name);
    const finite=values.map(v=>typeof v==='number'&&Number.isFinite(v)?v:null),present=finite.filter(v=>v!==null);
    const best=present.length>1?(higher?Math.max(...present):Math.min(...present)):null;
    finite.forEach(v=>tr.append(el('td',sig(v,scale),v!==null&&v===best?'best':'')));body.append(tr);
  }
  return table;
}
function renderMetrics(metrics){
  const card=$('model-metrics'),body=$('model-metrics-body'),caption=$('model-metrics-caption');if(!card||!body||!caption)return;
  const v2=metrics?.contact_v2;
  if(v2&&typeof v2==='object'){
    const engines=['old','new_raw','new_projected'],get=(...path)=>engines.map(e=>pick(v2,e,...path));
    const rows=[],capped=get('all','mean_penetration_capped_50mm_m').some(v=>typeof v==='number'&&Number.isFinite(v));
    for(const [key,label] of [['0.5s','0.5 s'],['1s','1 s'],['2s','2 s']])for(const regime of ['gentle','chaotic'])rows.push([`Position error @ ${label} · ${regime}`,'m',get(regime,'pos_err_m',key)]);
    rows.push(['Orientation error @ 2 s · gentle','deg',get('gentle','ori_err_deg_2s')],['Orientation error @ 2 s · chaotic','deg',get('chaotic','ori_err_deg_2s')],
      ['Fell / didn’t-fall accuracy','%',get('all','fell_accuracy'),100,true],capped?['Mean penetration depth (capped at 50 mm)','mm',get('all','mean_penetration_capped_50mm_m'),1000]:['Mean penetration depth','mm',get('all','mean_penetration_m'),1000],['Rest drift','mm/s',get('all','rest_drift_mps'),1000]);
    body.replaceChildren(metricsTable(['Old MLP','New (model only)','New + projection'],rows));
    const decision=v2.ship_decision||{},n=pick(v2,'old','all','n_episodes')??pick(v2,'new_projected','all','n_episodes'),total=pick(v2,'dataset','n_episodes'),gentle=pick(v2,'old','gentle','n_episodes'),chaotic=pick(v2,'old','chaotic','n_episodes'),seconds=pick(v2,'dataset','rollout_seconds');
    const shipped=typeof v2.shipped==='boolean'?(v2.shipped?`Shipped as default: ${decision.primary_variant==='new_raw'?'new Contact GNN (model only)':'new Contact GNN + projection'}.`:'Shipped as default: old MLP (Contact GNN is selectable as option C).'):typeof decision.ship==='boolean'?`Ship gate ${decision.ship?'passed':'not passed'}.`:'';
    const alone=typeof decision.model_alone_passes==='boolean'?` The learned model alone ${decision.model_alone_passes?'also passes':'does not pass'} the ship gate.`:'';
    caption.textContent=`${shipped?shipped+' ':''}“Projection” is a non-learned contact fix applied after every step; “model only” is the learned network alone.${alone} Held-out synthetic PyBullet scenes, n = ${n??DASH} episodes${total&&total!==n?` (${pick(v2,'dataset','subset')==='live_physics'?'default-physics ':''}subset of ${total} test episodes)`:''}${gentle!=null&&chaotic!=null?`: ${gentle} gentle / ${chaotic} chaotic`:''}; autoregressive${seconds?` ${seconds} s`:''} rollouts, never teacher-forced. Lower is better except accuracy; best per row highlighted. Not a guarantee on real photos.`;
    $('model-metrics-tag').textContent='HELD-OUT · SYNTHETIC';card.hidden=false;return;
  }
  const legacy=metrics?.mean_position_error_m;
  if(!legacy&&typeof metrics?.fell_agreement!=='number'){card.hidden=true;return;}
  body.replaceChildren(metricsTable([metrics.kind||'Current model'],[['Position error @ 0.5 s','m',[pick(legacy,'0.5')]],['Position error @ 1 s','m',[pick(legacy,'1.0')]],['Position error @ 2 s','m',[pick(legacy,'2.0')]],['Fell agreement with physics','%',[metrics.fell_agreement],100,true]]));
  caption.textContent=`Current shipped model only; no before/after comparison has been published yet. ${metrics.held_out_scenes??DASH} held-out synthetic scenes. ${metrics.note||''}`;
  $('model-metrics-tag').textContent='HELD-OUT · SYNTHETIC';card.hidden=false;
}
async function loadMetrics(){try{renderMetrics(await api('/api/metrics'));}catch(error){const card=$('model-metrics');if(card)card.hidden=true;console.warn('Model metrics unavailable:',error.message);}}
loadMetrics();
try{views.push(buildView('physics'),buildView('learned'));requestAnimationFrame(animate);await loadDemo();await scores();}catch(error){status(`Demo initialization failed: ${error.message}. WebGL is required for the 3D view; API docs at /docs.`);console.error(error);}
