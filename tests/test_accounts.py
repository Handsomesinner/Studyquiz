"""Account boundaries use synthetic provider replies and isolated SQLite only."""
import hashlib
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import accounts, main, security, store


@pytest.fixture
def account_env(tmp_path, monkeypatch):
    for name in ('VERCEL','VERCEL_ENV','AWS_LAMBDA_FUNCTION_NAME','TURSO_DATABASE_URL','LIBSQL_URL','STUDYQUIZ_TURSO_URL'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('STUDYQUIZ_DB',str(tmp_path/'accounts.db'))
    monkeypatch.setenv('STUDYQUIZ_AUTH_URL','https://synthetic.supabase.co')
    monkeypatch.setenv('STUDYQUIZ_AUTH_PUBLIC_KEY','synthetic-public-key')
    monkeypatch.setenv('STUDYQUIZ_AUTH_MODE','accounts')
    monkeypatch.setenv('STUDYQUIZ_ACCESS_PIN','legacy-code-that-must-not-bypass')
    store.reset_connection_state()
    def provider(path,payload=None,*,token=None):
        if path=='otp': return {}
        if path=='verify':
            if payload['token']!='123456': raise HTTPException(400,'Invalid code')
            return {'access_token':payload['email']}
        assert path=='user'
        return {'id':token, 'email':token,'email_confirmed_at':'2026-01-01'}
    monkeypatch.setattr(accounts,'provider',provider)
    yield
    store.reset_connection_state()


def sign_in(client,email='alice@example.test'):
    response=client.post('/api/account/verify',json={'email':email,'code':'123456'})
    assert response.status_code==200,response.text
    return response


def upload(client,name='private.txt'):
    response=client.post('/api/documents',files={'file':(name,b'Process scheduling uses a ready queue. '*20)})
    assert response.status_code==200,response.text
    return response.json()['id']


def test_verified_student_can_continue_across_devices_without_pin(account_env):
    first,second=TestClient(main.app),TestClient(main.app)
    reply=sign_in(first)
    assert 'httponly' in reply.headers['set-cookie'].lower()
    doc=upload(first)
    sign_in(second)
    assert second.get('/api/documents').json()[0]['id']==doc
    assert first.cookies.get(accounts.COOKIE)!=second.cookies.get(accounts.COOKIE)
    assert first.get('/api/account').json()['user']['email']=='alice@example.test'


def test_students_are_isolated_and_cannot_bypass_with_old_pin(account_env):
    alice,bob,guest=TestClient(main.app),TestClient(main.app),TestClient(main.app)
    sign_in(alice); doc=upload(alice)
    sign_in(bob,'bob@example.test')
    assert bob.get('/api/documents').json()==[]
    assert bob.delete('/api/documents/'+doc).status_code==404
    assert guest.get('/api/documents',headers={'X-StudyQuiz-Pin':'legacy-code-that-must-not-bypass'}).status_code==401
    guest.cookies.set(accounts.COOKIE,'a'*43)
    assert guest.get('/api/documents').status_code==401
    assert guest.get('/api/account/dashboard').status_code==401


def test_logout_revokes_session_and_logout_all_revokes_other_devices(account_env):
    one,two=TestClient(main.app),TestClient(main.app)
    sign_in(one);sign_in(two)
    stolen=one.cookies.get(accounts.COOKIE)
    assert one.post('/api/account/logout',json={}).status_code==200
    one.cookies.set(accounts.COOKIE,stolen)
    assert one.get('/api/documents').status_code==401
    sign_in(one)
    assert one.post('/api/account/logout-all',json={}).status_code==200
    assert two.get('/api/documents').status_code==401


def test_expired_sessions_and_unverified_email_fail_closed(account_env,monkeypatch):
    c=TestClient(main.app); sign_in(c)
    with store.connection() as conn: conn.execute('UPDATE student_sessions SET expires_at = ?', (int(time.time())-1,))
    assert c.get('/api/documents').status_code==401
    monkeypatch.setattr(accounts,'provider',lambda *args,**kwargs: {'access_token':'synthetic'} if args[0]=='verify' else {'id':'forged','email':'alice@example.test'})
    assert c.post('/api/account/verify',json={'email':'alice@example.test','code':'123456'}).status_code==401


def test_wrong_codes_and_cross_origin_requests_cannot_sign_in(account_env):
    c=TestClient(main.app)
    assert c.post('/api/account/verify',json={'email':'alice@example.test','code':'999999'}).status_code==400
    assert not c.cookies.get(accounts.COOKIE)
    assert c.post('/api/account/send-code',json={'email':'alice@example.test'},headers={'Origin':'https://attacker.test'}).status_code==403
    assert c.post('/api/account/send-code',json={'email':'invalid'}).status_code==422


def test_rate_limits_and_missing_configuration(account_env,monkeypatch):
    c=TestClient(main.app)
    for _ in range(4): assert c.post('/api/account/send-code',json={'email':'alice@example.test'}).status_code==200
    assert c.post('/api/account/send-code',json={'email':'alice@example.test'}).status_code==429
    monkeypatch.delenv('STUDYQUIZ_AUTH_PUBLIC_KEY')
    assert c.get('/api/documents').status_code==503
    assert c.post('/api/account/send-code',json={'email':'alice@example.test'}).status_code==503


def test_session_storage_contains_only_hashed_tokens(account_env):
    c=TestClient(main.app);sign_in(c)
    raw=c.cookies.get(accounts.COOKIE)
    with store.connection() as conn: row=conn.execute('SELECT * FROM student_sessions').fetchone()
    assert row['token_hash']==hashlib.sha256(raw.encode()).hexdigest()
    assert raw not in str(dict(row))


def test_native_integration_uses_public_key_never_service_role(account_env,monkeypatch):
    monkeypatch.delenv('STUDYQUIZ_AUTH_URL')
    monkeypatch.delenv('STUDYQUIZ_AUTH_PUBLIC_KEY')
    monkeypatch.setenv('NEXT_PUBLIC_SUPABASE_URL','https://synthetic.supabase.co')
    monkeypatch.setenv('SUPABASE_SERVICE_ROLE_KEY','must-not-use-this')
    monkeypatch.setenv('SUPABASE_PUBLISHABLE_KEY','synthetic-publishable')
    assert accounts.configured()
    assert accounts.public_key()=='synthetic-publishable'
    monkeypatch.delenv('SUPABASE_PUBLISHABLE_KEY')
    monkeypatch.delenv('SUPABASE_ANON_KEY',raising=False)
    monkeypatch.delenv('NEXT_PUBLIC_SUPABASE_ANON_KEY',raising=False)
    assert not accounts.configured()


def test_dashboard_is_private_and_counts_all_attempts(account_env):
    alice,bob=TestClient(main.app),TestClient(main.app)
    sign_in(alice);upload(alice,'Alice notes')
    sign_in(bob,'bob@example.test');upload(bob,'Bob notes')
    dash=alice.get('/api/account/dashboard').json()
    assert dash['document_count']==1
    assert [doc['title'] for doc in dash['documents']]==['Alice notes']
    assert dash['summary']=={'attempts':0,'correct':0,'answered':0,'percentage':None}
    owner=accounts.resolve_session(alice.cookies.get(accounts.COOKIE))['workspace']
    doc=dash['documents'][0]['id']
    with store.connection() as conn:
        conn.execute('INSERT INTO quizzes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',('quiz',doc,1,'Scheduling','[]','[]','{}','{}','[]','2026-01-01'))
        for i in range(25):
            aid=f'attempt-{i:02d}'
            conn.execute('INSERT INTO quiz_attempts VALUES (?, ?, ?, ?, ?, ?, ?)',(aid,'quiz','[]',1,2,'[]',f'2026-01-{i+1:02d}'))
            conn.execute('INSERT INTO resource_owners VALUES (?, ?, ?)',('attempt',aid,owner))
    dash=alice.get('/api/account/dashboard').json()
    assert dash['summary']['attempts']==25
    assert dash['summary']['percentage']==50
    assert len(dash['recent_attempts'])==20
    assert dash['needs_practice'][0]['topic']=='Scheduling'
    assert bob.get('/api/account/dashboard').json()['summary']['attempts']==0


def test_browser_import_is_explicit_and_cannot_claim_another_browser(account_env,monkeypatch):
    monkeypatch.setenv('STUDYQUIZ_AUTH_MODE','pin')
    old,other=TestClient(main.app),TestClient(main.app)
    for c in (old,other): c.headers['X-StudyQuiz-Pin']='legacy-code-that-must-not-bypass'
    mine=upload(old,'Mine');theirs=upload(other,'Theirs')
    sign_in(old)
    assert old.get('/api/documents').json()==[]
    assert old.get('/api/account').json()['browser_notes_to_import']==1
    assert old.post('/api/account/import-browser-notes',json={'previous_code':'wrong'}).status_code==401
    assert old.post('/api/account/import-browser-notes',json={'previous_code':'legacy-code-that-must-not-bypass'}).status_code==200
    assert [d['id'] for d in old.get('/api/documents').json()]==[mine]
    assert [d['id'] for d in other.get('/api/documents').json()]==[theirs]
