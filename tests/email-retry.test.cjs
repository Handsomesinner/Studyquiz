const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('app/static/index.html', 'utf8');
function setup(fail) {
  const nodes = new Map(); let now = 0, tick, requests = 0;
  const $ = id => {
    if (!nodes.has(id)) nodes.set(id, {value:'alice@example.test',disabled:false,textContent:'',classList:{remove(){}},focus(){}});
    return nodes.get(id);
  };
  const context = {$,Date:{now:()=>now},setInterval:fn=>(tick=fn,1),clearInterval(){},
    accountRequest:async()=>{requests++;if(fail){const e=new Error('Hourly code request limit');e.retryAfter=120;throw e;}return {message:'Check inbox'};}};
  vm.createContext(context);
  vm.runInContext(html.slice(html.indexOf('let emailRetryTimer;'),html.indexOf("$('email-verify-form').onsubmit")),context);
  return {$,context,advance(ms){now+=ms;tick();},requests:()=>requests};
}
test('rate-limited code requests show countdown and re-enable after retry time',async()=>{
  const s=setup(true);
  await s.$('email-signin-form').onsubmit({preventDefault(){}});
  assert.equal(s.$('send-email-code').disabled,true);
  assert.equal(s.$('send-email-code').textContent,'Try again in 2:00');
  s.advance(120000);
  assert.equal(s.$('send-email-code').disabled,false);
  assert.equal(s.$('send-email-code').textContent,'Email me a sign-in code');
  assert.equal(s.requests(),1);
});
test('successful code requests prevent accidental immediate resends',async()=>{
  const s=setup(false);
  await s.$('email-signin-form').onsubmit({preventDefault(){}});
  assert.equal(s.$('send-email-code').disabled,true);
  assert.equal(s.$('send-email-code').textContent,'Try again in 1:00');
  await s.$('email-signin-form').onsubmit({preventDefault(){}});
  assert.equal(s.requests(),1);
  s.advance(60000);
  assert.equal(s.$('send-email-code').disabled,false);
});
