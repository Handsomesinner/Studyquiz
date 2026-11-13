// Synthetic upload grants: no real credentials, network, or storage calls.
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');

async function request({pin='secret', configured=true, access='private', permitStatus=200, pathname=`uploads/${'a'.repeat(32)}-lecture.pdf`, type='blob.generate-client-token', method='POST', realSdk=false, accounts=false, cookie='sq_workspace=browser-token'}={}) {
  let issued=false, sdkCalls=0, permitCalls=0, forwarded;
  const process = {env: {STUDYQUIZ_ACCESS_PIN: configured ? 'secret' : '', BLOB_READ_WRITE_TOKEN: 'vercel_blob_rw_owned_synthetic' , BLOB_ACCESS: access, VERCEL_URL: 'studyquiz.example'}};
  if (accounts) Object.assign(process.env,{STUDYQUIZ_AUTH_MODE:'accounts',STUDYQUIZ_AUTH_URL:'https://synthetic.supabase.co',STUDYQUIZ_AUTH_PUBLIC_KEY:'synthetic'});
  const sandbox = {module: {exports: {}}, process, Buffer, Date, AbortSignal,
    require(name) {
      if(name==='node:crypto') return crypto;
      if(realSdk) return require('@vercel/blob/client');
      return {handleUpload: async (opts) => {
        sdkCalls++;
        if(type==='blob.upload-completed') throw new Error('invalid simulated callback signature');
        const grant=await opts.onBeforeGenerateToken(pathname);
        assert.equal(grant.addRandomSuffix, false);
        assert.equal(grant.allowOverwrite, false);
        issued=true;
        return {type, clientToken:'fake-client-token'};
      }};
    },
    fetch: async (url, opts) => { permitCalls++; forwarded={url, opts}; return {ok: permitStatus===200, status:permitStatus}; },
  };
  vm.runInNewContext(fs.readFileSync('api/blob-upload.js','utf8'), sandbox);
  const response={headers:{},setHeader(k,v){this.headers[k]=v}, status(n){this.statusCode=n;return this},json(body){this.body=body;return this}};
  await sandbox.module.exports({method,url:'/api/blob/upload',headers:{'x-studyquiz-pin':pin,cookie},body:{type,payload:{pathname,clientPayload:null,multipart:false}}},response);
  return {response,issued,sdkCalls,permitCalls,forwarded};
}

test('missing or wrong PIN cannot reach SDK token generation',async()=>{
  for(const pin of ['', 'wrong']) { const r=await request({pin}); assert.equal(r.response.statusCode,401); assert.equal(r.sdkCalls,0); assert.equal(r.issued,false); }
});
test('missing configuration fails closed',async()=>{assert.equal((await request({configured:false})).response.statusCode,401)});
test('public storage is rejected',async()=>{const r=await request({access:'public'}); assert.equal(r.response.statusCode,503); assert.equal(r.issued,false)});
test('valid request is budgeted and private',async()=>{const r=await request(); assert.equal(r.response.statusCode,200);assert.equal(r.response.body.access,'private');assert.equal(r.permitCalls,1);assert.equal(r.forwarded.url,'https://studyquiz.example/api/blob/permit');assert.equal(r.forwarded.opts.headers.cookie,'sq_workspace=browser-token')});
test('quota failure cannot issue a grant',async()=>{const r=await request({permitStatus:429});assert.equal(r.response.statusCode,429);assert.equal(r.issued,false)});
test('invalid extension is rejected',async()=>{const r=await request({pathname:`uploads/${'a'.repeat(32)}-script.exe`});assert.equal(r.response.statusCode,400);assert.equal(r.issued,false)});
test('unsigned completion callback is rejected by SDK boundary',async()=>{const r=await request({pin:'',type:'blob.upload-completed'});assert.equal(r.response.statusCode,400);assert.equal(r.issued,false)});

// SDK cryptography runs locally using a synthetic token; only the permit call is mocked.
test('installed SDK signs a private limited grant without real storage calls',async()=>{const r=await request({realSdk:true});assert.equal(r.response.statusCode,200);assert.equal(r.response.body.access,'private');assert.match(r.response.body.clientToken,/^vercel_blob_client_/);assert.equal(r.permitCalls,1)});
test('installed SDK rejects an unsigned completion callback',async()=>{const r=await request({realSdk:true,pin:'',type:'blob.upload-completed'});assert.equal(r.response.statusCode,400)});

test('accounts mode rejects the old PIN and verifies student cookies through the permit endpoint',async()=>{
  const denied=await request({accounts:true});assert.equal(denied.response.statusCode,401);
  const cookie='sq_student='+'a'.repeat(43);
  const forged=await request({accounts:true,cookie,pin:'',permitStatus:401});
  assert.equal(forged.issued,false);assert.equal(forged.response.statusCode,400);
  assert.equal(forged.forwarded.opts.headers['x-studyquiz-pin'],'');
  assert.equal(forged.forwarded.opts.headers.cookie,cookie);
  const accepted=await request({accounts:true,cookie,pin:''});
  assert.equal(accepted.issued,true);assert.equal(accepted.response.statusCode,200);
});
