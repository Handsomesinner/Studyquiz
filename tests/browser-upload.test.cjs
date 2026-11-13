const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('browser PUT uses only Blob client headers accepted by CORS', async () => {
  let request;
  const allowed = new Set(['authorization', 'x-api-version', 'x-vercel-blob-access', 'x-content-type', 'x-content-length']);
  class XHR {
    constructor() { this.upload = {}; this.headers = {}; request = this; }
    open(method, url) { this.method = method; this.url = url; }
    setRequestHeader(name, value) {
      assert.ok(allowed.has(name), `Unsupported browser upload header: ${name}`);
      this.headers[name] = value;
    }
    send(file) {
      this.file = file; this.status = 200;
      this.responseText = JSON.stringify({url: 'https://synthetic.private.blob.vercel-storage.com/uploads/test.txt'});
      this.onload();
    }
  }
  const html = fs.readFileSync('app/static/index.html', 'utf8');
  const code = html.slice(html.indexOf('function xhrPutToBlob('), html.indexOf('\n/**\n * Native Blob client upload'));
  const context = vm.createContext({XMLHttpRequest: XHR, URLSearchParams});
  vm.runInContext(code, context);
  const file = {size: 9};
  const result = await context.xhrPutToBlob(file, {pathname:'uploads/test.txt', contentType:'text/plain',clientToken:'synthetic',access:'private'});
  assert.equal(request.method, 'PUT');
  assert.equal(new URL(request.url).origin, 'https://vercel.com');
  assert.equal(request.headers['x-vercel-blob-access'], 'private');
  assert.equal(request.headers.authorization, 'Bearer synthetic');
  assert.equal(request.file, file);
  assert.match(result.url, /private\.blob/);
});
