const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(status=200) {
  const elements = new Map(), stored = new Map();
  let requests=0, refreshed=0;
  const element = id => {
    if(!elements.has(id)) elements.set(id, {value:'',textContent:'',disabled:false,hidden:true,focus(){this.focused=true},scrollIntoView(){},classList:{remove(){element(id).hidden=false},add(){element(id).hidden=true}}});
    return elements.get(id);
  };
  const context = {document:{getElementById:element},sessionStorage:{getItem:k=>stored.get(k)||'',setItem:(k,v)=>stored.set(k,v),removeItem:k=>stored.delete(k)},
    window:{prompt(){throw new Error('Native prompt must never be used for access codes')}},
    fetch:async(url,options)=>{requests++;assert.equal(url,'/api/documents');assert.equal(options.headers['X-StudyQuiz-Pin'],element('access-code').value);return {ok:status===200,status,json:async()=>[]}},
    authReady:Promise.resolve(),refreshDocumentList:async()=>{refreshed++},errorDetail:()=> 'Server setup error'};
  vm.createContext(context);
  const html=fs.readFileSync('app/static/index.html','utf8');
  const code=html.slice(html.indexOf('const PIN_STORAGE_KEY'),html.indexOf('\nasync function api('));
  vm.runInContext('const $ = id => document.getElementById(id);\n'+code,context);
  return {element,stored,context,requests:()=>requests,refreshed:()=>refreshed};
}

test('missing access code shows an inline password form without a popup',()=>{
  const s=setup();
  assert.throws(()=>vm.runInContext('ensureAccessPinInteractive()',s.context),/Unlock your workspace/);
  assert.equal(s.element('access-card').hidden,false);
  assert.equal(s.element('access-code').focused,true);
  assert.equal(s.requests(),0);
});
test('incorrect code is not stored and the password field is cleared',async()=>{
  const s=setup(401); s.element('access-code').value='synthetic-wrong';
  await s.element('access-form').onsubmit({preventDefault(){}});
  assert.equal(s.stored.size,0);assert.equal(s.element('access-code').value,'');
  assert.match(s.element('access-status').textContent,/not accepted/);
  assert.equal(s.element('access-unlock').disabled,false);
});
test('accepted code is tab-scoped and clears the visible input',async()=>{
  const s=setup();s.element('access-code').value='synthetic-valid-code';
  await s.element('access-form').onsubmit({preventDefault(){}});
  assert.equal(s.stored.get('sq-access-pin'),'synthetic-valid-code');
  assert.equal(s.element('access-code').value,'');assert.equal(s.element('access-card').hidden,true);
  assert.equal(s.refreshed(),1);
});
test('blank code does not send a request',async()=>{
  const s=setup();await s.element('access-form').onsubmit({preventDefault(){}});
  assert.equal(s.requests(),0);assert.equal(s.stored.size,0);
});
