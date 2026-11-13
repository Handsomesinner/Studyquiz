"""HTTP boundary regressions: synthetic data, isolated DB, no paid API or uploads."""
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import generator, grounding, main, pdf_processor, security, store
from app.retriever import BM25Retriever


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for key in ('VERCEL', 'VERCEL_ENV', 'AWS_LAMBDA_FUNCTION_NAME', 'TURSO_DATABASE_URL', 'LIBSQL_URL', 'STUDYQUIZ_TURSO_URL', 'ACCESS_PIN', 'STUDYQUIZ_API_SECRET'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('STUDYQUIZ_ACCESS_PIN', 'test-pin')
    monkeypatch.setenv('STUDYQUIZ_DB', str(tmp_path / 'isolated.db'))
    store.reset_connection_state()
    yield
    store.reset_connection_state()


def client():
    c = TestClient(main.app)
    c.get('/api/health')  # establishes an independent browser workspace
    c.headers['X-StudyQuiz-Pin'] = 'test-pin'
    return c


def upload(c, title='lecture.txt'):
    r = c.post('/api/documents', files={'file': (title, b'The round-robin algorithm uses a fixed time quantum. ' * 10)})
    assert r.status_code == 200, r.text
    return r.json()['id']


def seed_paper(c, doc_id):
    owner = c.get('/api/workspace').json()['workspace_id']
    token = security.workspace.set(owner)
    try:
        store.save_exam_paper(exam_id='abcdef123456', document_id=doc_id, use_rag=True, topic=None,
                              difficulty='medium', paper={'questions': []}, context_chunks=[])
    finally:
        security.workspace.reset(token)
    return 'abcdef123456'


def test_pin_protects_answers_exports_and_drafts(sandbox):
    c = client()
    c.headers.pop('X-StudyQuiz-Pin')
    for method, path in [('GET', '/api/documents'), ('GET', '/api/evaluation/export'),
                         ('GET', '/api/quiz/abc/evaluation'), ('GET', '/api/exam/abc/draft'),
                         ('PUT', '/api/exam/abc/draft')]:
        assert c.request(method, path, json={'answers': {}}).status_code == 401


def test_documents_are_browser_private_even_with_same_pin(sandbox):
    alice, bob = client(), client()
    doc = upload(alice)
    assert len(alice.get('/api/documents').json()) == 1
    assert bob.get('/api/documents').json() == []
    assert bob.delete('/api/documents/' + doc).status_code == 404
    assert bob.post('/api/quiz', json={'document_id': doc, 'num_questions': 1}).status_code == 404
    assert alice.get('/api/documents').headers['cache-control'] == 'no-store'


def test_password_share_grant_drafts_isolated_and_revocable(sandbox):
    owner, alice, bob = client(), client(), client()
    exam = seed_paper(owner, upload(owner))
    shared = owner.post('/api/share', json={'resource_type': 'exam', 'resource_id': exam, 'password': 'practice-only'})
    assert shared.status_code == 200, shared.text
    token = shared.json()['token']
    alice.headers.pop('X-StudyQuiz-Pin')
    bob.headers.pop('X-StudyQuiz-Pin')
    assert alice.get(f'/api/exam/{exam}').status_code == 401
    assert alice.post(f'/api/share/{token}/redeem', json={'password': 'wrong'}).status_code == 403
    for c in (alice, bob):
        assert c.post(f'/api/share/{token}/redeem', json={'password': 'practice-only'}).status_code == 200
        assert c.get(f'/api/exam/{exam}').status_code == 200
    a = alice.put(f'/api/exam/{exam}/draft', json={'answers': {'1': 'Alice private answer'}})
    assert a.status_code == 200, a.text
    assert bob.get(f'/api/exam/{exam}/draft').json()['answers'] == {}
    assert bob.put(f'/api/exam/{exam}/draft', json={'attempt_id': a.json()['attempt_id'], 'answers': {'1': 'overwrite'}}).status_code == 404
    assert alice.get(f'/api/exam/{exam}/draft').json()['answers']['1'] == 'Alice private answer'
    assert bob.post(f'/api/exam/{exam}/answers').status_code == 401
    assert owner.delete(f'/api/share/{token}').status_code == 200
    assert alice.get(f'/api/exam/{exam}').status_code == 401


def test_public_deployment_fails_closed(sandbox, monkeypatch):
    c = client()
    monkeypatch.setenv('VERCEL', '1')
    monkeypatch.delenv('STUDYQUIZ_ACCESS_PIN')
    assert c.get('/api/documents').status_code == 503
    monkeypatch.setenv('STUDYQUIZ_ACCESS_PIN', 'strong-test-pin-at-least-sixteen')
    c.headers['X-StudyQuiz-Pin'] = 'strong-test-pin-at-least-sixteen'
    assert c.get('/api/documents').status_code == 503  # no durable database
    assert security.pin_matches(None) is False


def test_budget_is_atomic_across_workers(sandbox):
    def reserve(_):
        return store.consume_budget('same-day-bucket', 3)
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(reserve, range(12))) == 3


def test_baseline_not_forced_zero_and_duplicate_options_rejected():
    q = generator.QuizQuestion(question='What does RR use?', options=['Quantum', 'Priority', 'Stack', 'Queue'], correct_index=0,
                              explanation='From notes', source_quote='Round robin uses a fixed time quantum.')
    source = q.source_quote
    g = grounding.validate_question(q, source_text=source, use_rag=False)
    assert g.grounded is True
    assert g.expected_grounded is False
    q.options = ['Quantum', 'quantum', 'Stack', 'Queue']
    g = grounding.validate_question(q, source_text=source, use_rag=True)
    assert grounding.filter_grounded([q], [g])[0] == []


def test_strict_failure_recorded_and_no_questions_served(sandbox, monkeypatch):
    c = client()
    doc = upload(c)
    q = generator.QuizQuestion(question='What is the invented answer?', options=['A', 'B', 'C', 'D'], correct_index=0,
                              explanation='Not in notes', source_quote='Unrelated invented quotation is not present.')
    def fake(**_):
        gs, metrics = grounding.validate_quiz([q], source_text='actual lecture text', use_rag=True)
        return generator.Quiz(questions=[q]), gs, metrics
    monkeypatch.setattr(main, '_generate_mcq_attempt', fake)
    r = c.post('/api/quiz', json={'document_id': doc, 'num_questions': 1, 'use_rag': True, 'require_grounding': True})
    assert r.status_code == 422, r.text
    summary = c.get('/api/evaluation/summary').json()
    assert summary['accuracy_comparison_available'] is False
    run = summary['generation_runs'][0]
    assert run['status'] == 'verification_failed'
    assert len(run['details']['attempts']) == 2
    assert summary['by_condition']['rag']['question_count'] == 1  # original, not retry-selected output


def test_url_auth_header_never_goes_to_custom_or_other_store(sandbox, monkeypatch):
    monkeypatch.setenv('BLOB_READ_WRITE_TOKEN', 'vercel_blob_rw_owned_secret')
    assert security.is_blob_url_for_auth_header('https://owned.private.blob.vercel-storage.com/a.pdf')
    assert not security.is_blob_url_for_auth_header('https://other.private.blob.vercel-storage.com/a.pdf')
    monkeypatch.setenv('BLOB_ALLOWED_HOSTS', 'example.org')
    assert not security.is_blob_url_for_auth_header('https://example.org/a.pdf')
    with pytest.raises(ValueError):
        security.validate_download_url('https://owned.blob.vercel-storage.com:8443/a.pdf')


def test_upload_permit_scoped_and_cleanup_on_extraction_error(sandbox, monkeypatch):
    alice, bob = client(), client()
    monkeypatch.setenv('BLOB_READ_WRITE_TOKEN', 'vercel_blob_rw_owned_secret')
    path = 'uploads/' + 'a' * 32 + '-lecture.pdf'
    assert alice.post('/api/blob/permit', json={'pathname': path}).status_code == 200
    url = 'https://owned.private.blob.vercel-storage.com/' + path
    assert bob.post('/api/documents', json={'url': url}).status_code == 404
    calls = []
    monkeypatch.setattr(main, '_download_url_to_bytes', lambda _: b'invalid pdf')
    monkeypatch.setattr(main, '_delete_uploaded_source', lambda pathname: calls.append(pathname) or True)
    assert alice.post('/api/documents', json={'url': url, 'filename': 'lecture.pdf'}).status_code == 400
    assert calls == [path]


def test_csv_formula_cells_neutralized(sandbox, monkeypatch):
    c = client()
    monkeypatch.setattr(store, 'list_eval_rows', lambda: [{'document_title': '=1+1', 'question': ' +1', 'source_quote': '@SUM(A1)'}])
    r = c.get('/api/evaluation/export')
    assert r.status_code == 200
    assert "'=1+1" in r.text and "' +1" in r.text and "'@SUM(A1)" in r.text


def test_retrieval_empty_terms_and_single_sample():
    r = BM25Retriever()
    r.index(['the and of', '!!!'])
    assert r.search('topic') == []
    assert len(r.spread_sample(1)) == 1
    assert r.search('topic', top_k=0) == []


def test_office_archive_expansion_rejected_before_parser():
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('word/document.xml', b'x' * (2 * 1024 * 1024))
    with pytest.raises(ValueError, match='compressed'):
        pdf_processor.extract_text_with_meta('lecture.docx', data.getvalue())


def test_model_budget_limits_calls_and_context_can_follow_workers(sandbox, monkeypatch):
    from contextvars import copy_context
    from fastapi import HTTPException
    monkeypatch.setenv('AI_DAILY_GLOBAL_CALLS', '5')
    monkeypatch.setenv('AI_DAILY_WORKSPACE_CALLS', '1')
    t = security.workspace.set('test-workspace')
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(copy_context().run, security.reserve_model_call).result()
        with pytest.raises(HTTPException) as exc:
            security.reserve_model_call()
        assert exc.value.status_code == 429
    finally:
        security.workspace.reset(t)


def test_human_reviews_are_separate_from_quote_metrics(sandbox):
    c = client()
    doc = upload(c)
    owner = c.get('/api/workspace').json()['workspace_id']
    t = security.workspace.set(owner)
    try:
        q = generator.QuizQuestion(question='Q?', options=['A', 'B', 'C', 'D'], correct_index=0,
                                  explanation='Explanation', source_quote='A quoted source sentence appears in notes.')
        gs, m = grounding.validate_quiz([q], source_text=q.source_quote, use_rag=True)
        store.save_quiz(quiz_id='abcd12', document_id=doc, use_rag=True, topic=None, questions=[q], groundings=gs,
                        pre_filter_metrics=m.to_dict(), served_metrics=m.to_dict(), context_chunks=[])
        store.append_eval_rows(main._build_eval_rows(quiz_id='abcd12', doc_id=doc, doc_title='Lecture', topic=None,
                                                    use_rag=True, questions=[q], groundings=gs, phase='served'))
    finally:
        security.workspace.reset(t)
    review = c.put('/api/quiz/abcd12/reviews/0', json={'answer_correct': False, 'evidence_supports': False, 'clear_question': True})
    assert review.status_code == 200
    summary = c.get('/api/evaluation/summary').json()
    assert summary['human_review']['rag']['answer_correct_rate'] == 0
    assert summary['human_review']['baseline']['answer_correct_rate'] is None
    assert c.get('/api/evaluation/export').status_code == 200
    assert client().put('/api/quiz/abcd12/reviews/0', json={'answer_correct': True, 'evidence_supports': True, 'clear_question': True}).status_code == 404
    assert c.delete('/api/documents/' + doc).status_code == 200
    assert c.get('/api/evaluation/summary').json()['quizzes'] == 0


def test_source_cleanup_failure_is_retained_for_retry(sandbox, monkeypatch):
    c = client()
    doc = upload(c)
    path = 'uploads/' + 'b' * 32 + '-lecture.pdf'
    owner = c.get('/api/workspace').json()['workspace_id']
    with store.connection() as conn:
        conn.execute('INSERT INTO upload_permits VALUES (?, ?, ?)', (path, owner, '2020-01-01T00:00:00+00:00'))
        conn.execute('INSERT INTO document_sources VALUES (?, ?)', (doc, path))
    monkeypatch.setattr(main, '_delete_uploaded_source', lambda _: False)
    assert c.delete('/api/documents/' + doc).status_code == 503
    assert len(c.get('/api/documents').json()) == 1
    assert c.post('/api/uploads/cleanup').json()['retry_pending'] == 1


def test_redirect_to_other_store_never_gets_authorization(sandbox, monkeypatch):
    c = client()
    monkeypatch.setenv('BLOB_READ_WRITE_TOKEN', 'vercel_blob_rw_owned_secret')
    pathname = 'uploads/' + 'c' * 32 + '-lecture.pdf'
    c.post('/api/blob/permit', json={'pathname': pathname})
    url = 'https://owned.private.blob.vercel-storage.com/' + pathname
    seen = []
    class FakeResponse:
        status_code = 302
        headers = {'location': 'https://other.private.blob.vercel-storage.com/elsewhere.pdf'}
        def __enter__(self): return self
        def __exit__(self, *_): pass
    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def stream(self, method, url): seen.append(url); return FakeResponse()
    monkeypatch.setattr(main.httpx, 'Client', FakeClient)
    monkeypatch.setattr(main, '_delete_uploaded_source', lambda _: True)
    # Existing TestClient holds its transport; only the remote downloader is replaced.
    r = c.post('/api/documents', json={'url': url, 'filename': 'lecture.pdf'})
    assert r.status_code == 400
    assert seen == [url]


def test_unclaimed_legacy_attempt_cannot_be_claimed_by_supplied_id(sandbox):
    c = client()
    doc = upload(c)
    exam = seed_paper(c, doc)
    # Synthetic pre-upgrade row has no resource_owners entry.
    store.save_exam_attempt(attempt_id='legacy-attempt', exam_id=exam, answers={'1': 'private legacy answer'})
    r = c.put(f'/api/exam/{exam}/draft', json={'attempt_id': 'legacy-attempt', 'answers': {'1': 'hijack'}})
    assert r.status_code == 404
    assert store.get_exam_attempt('legacy-attempt')['answers']['1'] == 'private legacy answer'


def test_complete_generation_failure_is_visible_without_evaluation_rows(sandbox, monkeypatch):
    c = client()
    doc = upload(c)
    def fail(**kwargs):
        raise generator.GenerationError("synthetic failure")
    monkeypatch.setattr(main, '_generate_mcq_attempt', fail)
    r = c.post('/api/quiz', json={'document_id': doc, 'num_questions': 1})
    assert r.status_code == 503
    summary = c.get('/api/evaluation/summary').json()
    assert summary['quizzes'] == 0
    assert summary['generation_runs'][0]['status'] == 'generation_failed'


def test_public_missing_configuration_does_not_crash_lifespan(sandbox, monkeypatch):
    monkeypatch.setenv('VERCEL', '1')
    monkeypatch.delenv('STUDYQUIZ_ACCESS_PIN')
    with TestClient(main.app) as c:
        assert c.get('/').status_code == 200
        health = c.get('/api/health').json()
        assert health['status'] == 'degraded'
        assert c.get('/api/documents').status_code == 503


def test_private_store_credential_takes_precedence_over_legacy_public(sandbox, monkeypatch):
    monkeypatch.setenv('BLOB_READ_WRITE_TOKEN', 'vercel_blob_rw_legacy_synthetic')
    monkeypatch.setenv('STUDYQUIZ_PRIVATE_BLOB_READ_WRITE_TOKEN', 'vercel_blob_rw_private_synthetic')
    assert security.is_blob_url_for_auth_header('https://private.private.blob.vercel-storage.com/notes.pdf')
    assert not security.is_blob_url_for_auth_header('https://legacy.public.blob.vercel-storage.com/notes.pdf')
