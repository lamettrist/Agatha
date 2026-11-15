"""Optional real-browser QA: pip install playwright; playwright install chromium.

Use --chromium for an existing headless shell and --photo for a real image fixture.
"""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://127.0.0.1:8017')
    parser.add_argument('--chromium');parser.add_argument('--photo',type=Path);args=parser.parse_args()
    with sync_playwright() as p:
        options=dict(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
        if args.chromium: options['executable_path']=args.chromium
        browser=p.chromium.launch(**options);page=browser.new_page(viewport={'width':1440,'height':1100})
        errors=[];requests=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda request:requests.append(request.url))
        page.goto(args.base);page.wait_for_function("document.getElementById('status').textContent.startsWith('Ready.')")
        page.wait_for_timeout(400)
        point=page.evaluate('''async ()=>{
          const THREE=await import('/static/vendor/three.module.js');
          const r=document.querySelector('#physics-view canvas').getBoundingClientRect();
          const camera=new THREE.PerspectiveCamera(43,r.width/r.height,.01,40);
          camera.position.set(1.55,1.6,1.85);camera.lookAt(0,.63,0);camera.updateMatrixWorld();
          const v=new THREE.Vector3(.49,.805,.16).project(camera);
          return {x:r.x+(v.x+1)/2*r.width,y:r.y+(-v.y+1)/2*r.height};
        }''')
        page.mouse.click(point['x'],point['y'])
        page.wait_for_function("document.getElementById('status').textContent.startsWith('Physics fell:')")
        assert page.locator('#object').input_value()=='mug'
        assert page.locator('#fell').inner_text()=='1'
        page.wait_for_function("document.getElementById('time').textContent.startsWith('3.00')")
        print('Click-to-push / playback:',page.locator('#status').inner_text(),page.locator('#error').inner_text(),flush=True)
        page.screenshot(path='docs/demo.png',full_page=True)
        page.select_option('#engine','learned');assert not page.locator('#physics-panel').is_visible()
        page.select_option('#engine','both')
        page.fill('#name','Browser QA');page.click('#save-score')
        page.wait_for_function("document.getElementById('status').textContent==='Physics score submitted.'")
        page.click('#ask');page.wait_for_function("!document.getElementById('ask').disabled",timeout=120000)
        assert page.locator('#answer').inner_text()
        print('Engine toggle / leaderboard / assistant response: passed',flush=True)
        page.click('#load-bottle');page.wait_for_function("document.getElementById('object').value==='bottle'")
        for kind in ('mlp_multistep','graph_impulse'):
            page.select_option('#architecture',kind)
            with page.expect_response(lambda r:r.url.endswith('/api/simulate')) as response:
                page.click('#run')
            prediction=response.value.json()
            assert prediction['learned']['model_architecture']==kind
            assert max(f['objects'][0]['position'][1] for f in prediction['physics']['frames'])>1.2
            page.wait_for_function("!document.getElementById('run').disabled")
        page.select_option('#architecture','mlp_multistep');page.uncheck('#table-collisions')
        with page.expect_response(lambda r:r.url.endswith('/api/simulate')) as response:
            page.click('#run')
        prediction=response.value.json()
        assert prediction['physics']['support_collisions_enabled'] is False
        assert prediction['learned']['support_collisions_enabled'] is False
        assert prediction['physics']['events']['fell']==['bottle']
        page.wait_for_function("document.getElementById('time').textContent.startsWith('3.00')")
        page.screenshot(path='docs/bottle-noclip.png',full_page=True)
        print('Bottle toss / both learned architectures / deliberately disabled table: passed',flush=True)
        if args.photo:
            page.set_input_files('#photo',str(args.photo))
            page.wait_for_function("document.getElementById('reconstruction').textContent.includes('detected ·')",timeout=90000)
            page.locator('#photo-wrap').wait_for(state='visible',timeout=15000)
            assert '0 kept' not in page.locator('#reconstruction').inner_text()
            print('Real photo upload:',page.locator('#reconstruction').inner_text(),flush=True)
            page.click('#run');page.wait_for_function("document.getElementById('status').textContent.startsWith('Physics fell:')")
            print('Real photo simulation:',page.locator('#status').inner_text(),flush=True)
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(400)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),'Mobile horizontal overflow'
        assert not errors,errors
        assert all(url.startswith(args.base) or url.startswith(('blob:','data:')) for url in requests),requests
        print('Mobile overflow / no external browser requests / no JS errors: passed',flush=True)
        browser.close()

if __name__=='__main__':main()
