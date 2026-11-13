const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('each question has a unique accessible name and an independent radio group', () => {
  const elements = [];
  const createElement = tag => {
    const el = {tag, children: [], attributes: {}, dataset: {}, listeners: {},
      appendChild(child) { this.children.push(child); },
      setAttribute(name,value) { this.attributes[name]=value; },
      addEventListener(name,handler) { this.listeners[name]=handler; },
      classList: {remove(){}}};
    elements.push(el); return el;
  };
  const holder = createElement('div'); let progress=0;
  const context=vm.createContext({document:{createElement},$:()=>holder,updateProgress(){progress++}});
  const html=fs.readFileSync('app/static/index.html','utf8');
  vm.runInContext(html.slice(html.indexOf('function renderQuiz('),html.indexOf('\nfunction renderExamPaper(')),context);
  context.renderQuiz([{question:'First question?',options:['One','Two']},{question:'Second question?',options:['Three','Four']}]);
  const ids=new Map(elements.filter(e=>e.id).map(e=>[e.id,e.textContent]));
  const groups=holder.children;
  assert.equal(groups.length,2);
  groups.forEach((group,i)=>{
    assert.equal(group.attributes.role,'radiogroup');
    const name=group.attributes['aria-labelledby'].split(' ').map(id=>ids.get(id)).join(' ');
    assert.ok(name.includes(`Question ${i+1} of 2`));
    assert.ok(name.includes(i===0?'First question?':'Second question?'));
    const labels=group.children.filter(e=>e.tag==='label');
    labels.forEach(label=>{
      assert.equal(label.children[0].name,`q${i}`);
      assert.ok(label.children[1].textContent);
      label.children[0].listeners.change();
    });
  });
  assert.equal(progress,5);
});
