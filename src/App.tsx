import { useEffect, useRef, useState } from 'react';
import { AlertCircle, Brain, FileText, Loader2, Play, RefreshCw, Upload, X } from 'lucide-react';

type Result = {
  filename?: string;
  parsed: { name: string; email: string; phone: string; links: string[]; sections: Record<string, string>; text: string };
  classification: { track: string; confidence: number; probabilities: Record<string, number> };
  match: { hybrid_score: number; semantic: number; skill_overlap: number; track_fit: number; matched: string[]; missing: string[]; extra: string[]; backend: string };
};
type Cand = { id: string; label: string; file?: File; text?: string; result?: Result; error?: string };

const JDS: Record<string, string> = {
  'Senior Frontend Engineer': 'We are hiring a Senior Frontend Engineer to build our web platform. You will work with React, TypeScript, JavaScript, HTML and CSS, consume REST API and GraphQL services, write unit testing, and collaborate in an agile team using git.',
  'MLOps Engineer': 'Looking for an MLOps Engineer to deploy machine learning models to production. Requires Python, docker, kubernetes, ci/cd, pytorch, airflow, monitoring and data pipelines on aws. Experience with terraform is a plus.',
  'Security Analyst': 'Join our SOC as a Security Analyst. You will run incident response, vulnerability assessment and threat modeling, operate a siem such as splunk, follow owasp guidance, and support iso 27001 compliance.',
};
const SAMPLES: Cand[] = [
  { id: 's1', label: 'Frontend Lead', text: 'Minh Tran\nminh.tran@example.com | +84 912 345 678 | github.com/minhtran\nSummary\nFrontend Lead with 8 years of experience.\nExperience\n- Led a team building a design system with React and TypeScript\n- Built REST API clients, GraphQL queries and unit testing with agile practices\nSkills\nreact, typescript, javascript, html, css, git, node.js, graphql\nEducation\nBSc Computer Science' },
  { id: 's2', label: 'MLOps Engineer', text: 'Lan Nguyen\nlan.nguyen@example.com | +84 987 654 321\nSummary\nMLOps Engineer with 5 years of experience deploying models.\nExperience\n- Trained pytorch models and shipped them with docker and kubernetes\n- Built airflow data pipelines and ci/cd on aws, added monitoring with prometheus\nSkills\npython, pytorch, docker, kubernetes, airflow, aws, mlops, machine learning, ci/cd\nEducation\nMSc Data Science' },
  { id: 's3', label: 'Security Analyst', text: 'An Pham\nan.pham@example.com | +84 905 111 222\nSummary\nSecurity Analyst with 4 years in a SOC.\nExperience\n- Led incident response and vulnerability assessment using splunk and siem tooling\n- Applied owasp guidance and supported iso 27001 audits and compliance\nSkills\nsiem, splunk, owasp, incident response, threat modeling, nmap, compliance, firewall\nEducation\nBSc Information Security' },
];
const PHASES = ['Đang trích xuất văn bản', 'Đang vector hoá', 'Đang tính embedding ngữ nghĩa', 'Đang tổng hợp điểm hybrid'];
const tone = (s: number) => (s > 75 ? 'text-emerald-400' : s >= 50 ? 'text-amber-400' : 'text-rose-400');
const bar = (s: number) => (s > 75 ? 'bg-emerald-500' : s >= 50 ? 'bg-amber-500' : 'bg-rose-500');
const num = 'font-mono tabular-nums';

function Meter({ label, value }: { label: string; value: number }) {
  return (
    <div className="min-w-24">
      <div className="flex justify-between text-xs text-zinc-400"><span>{label}</span><span className={num}>{value.toFixed(0)}%</span></div>
      <div className="mt-1 h-1.5 bg-[#334155]"><div className={`h-full ${bar(value)}`} style={{ width: `${Math.min(value, 100)}%` }} /></div>
    </div>
  );
}

export default function App() {
  const [jdKey, setJdKey] = useState(Object.keys(JDS)[0]);
  const [jd, setJd] = useState(JDS[jdKey]);
  const [cands, setCands] = useState<Cand[]>([]);
  const [paste, setPaste] = useState('');
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState(0);
  const [open, setOpen] = useState<Cand | null>(null);
  const [info, setInfo] = useState<any>(null);
  const [training, setTraining] = useState(false);
  const [testText, setTestText] = useState('');
  const [testOut, setTestOut] = useState<string>('');
  const input = useRef<HTMLInputElement>(null);

  const loadInfo = () => fetch('/api/model-info').then(r => r.json()).then(setInfo).catch(() => setInfo(null));
  useEffect(() => { loadInfo(); }, []);
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => setPhase(p => (p + 1) % PHASES.length), 900);
    return () => clearInterval(t);
  }, [busy]);

  const addFiles = (fs: FileList | null) => {
    if (!fs) return;
    const add: Cand[] = [];
    Array.from(fs).forEach(f => {
      if (f.size > 10 * 1024 * 1024) add.push({ id: crypto.randomUUID(), label: f.name, error: 'Tệp vượt quá 10MB.' });
      else add.push({ id: crypto.randomUUID(), label: f.name, file: f });
    });
    setCands(c => [...c, ...add]);
  };
  const addPaste = () => { if (paste.trim().length >= 20) { setCands(c => [...c, { id: crypto.randomUUID(), label: 'CV dán vào', text: paste }]); setPaste(''); } };

  const screen = async (list: Cand[]) => {
    setBusy(true); setPhase(0);
    const out = await Promise.all(list.map(async c => {
      if (c.error && !c.file && !c.text) return c;
      const fd = new FormData(); fd.append('jd_text', jd);
      c.file ? fd.append('file', c.file) : fd.append('resume_text', c.text ?? '');
      try {
        const r = await fetch('/api/screen-resume', { method: 'POST', body: fd });
        const j = await r.json();
        if (!r.ok) return { ...c, result: undefined, error: typeof j.detail === 'string' ? j.detail : 'Dữ liệu không hợp lệ.' };
        return { ...c, result: j as Result, error: undefined };
      } catch { return { ...c, error: 'Không kết nối được máy chủ.' }; }
    }));
    setCands(out); setBusy(false);
  };
  const ranked = [...cands].sort((a, b) => (b.result?.match.hybrid_score ?? -1) - (a.result?.match.hybrid_score ?? -1));

  const retrain = async () => { setTraining(true); await fetch('/api/train', { method: 'POST' }).catch(() => {}); await loadInfo(); setTraining(false); };
  const runTest = async () => {
    const r = await fetch('/api/match-job', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ resume_text: testText, jd_text: jd }) });
    const j = await r.json();
    setTestOut(r.ok ? `${j.classification.track} (${(j.classification.confidence * 100).toFixed(1)}%)` : String(j.detail?.[0]?.msg ?? j.detail));
  };

  const meta = info?.metadata;
  return (
    <div className="min-h-screen font-sans">
      <header className="border-b border-[#334155] bg-[#1E293B] px-6 py-3 flex flex-wrap items-center justify-between gap-3">
        <div><div className="text-xs text-zinc-400">Tuyển dụng / Sàng lọc</div><h1 className="text-lg font-semibold">{jdKey}</h1></div>
        <button disabled={busy || !cands.length} onClick={() => screen(cands)} className="flex items-center gap-2 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 text-slate-950 font-semibold px-4 py-2 rounded-sm">
          {busy ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />} Sàng lọc {cands.length || ''} ứng viên
        </button>
      </header>

      {busy && <div className="bg-[#1E293B] border-b border-[#334155] px-6 py-2 flex gap-4 text-xs overflow-x-auto" role="status">
        {PHASES.map((p, i) => <span key={p} className={i === phase ? 'text-emerald-400' : 'text-zinc-500'}>{p}</span>)}
      </div>}

      <main className="grid lg:grid-cols-[380px_1fr] gap-px bg-[#334155]">
        <section className="bg-[#0F172A] p-5 space-y-5">
          <div>
            <h2 className="font-semibold mb-2">Mô tả công việc</h2>
            <select value={jdKey} onChange={e => { setJdKey(e.target.value); setJd(JDS[e.target.value]); }} className="w-full bg-[#1E293B] border border-[#334155] p-2 mb-2 text-sm">
              {Object.keys(JDS).map(k => <option key={k}>{k}</option>)}
            </select>
            <textarea value={jd} onChange={e => setJd(e.target.value)} rows={6} className="w-full bg-[#1E293B] border border-[#334155] p-2 text-sm" aria-label="Nội dung JD" />
          </div>
          <div>
            <h2 className="font-semibold mb-2">Hồ sơ ứng viên</h2>
            <div onClick={() => input.current?.click()} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); addFiles(e.dataTransfer.files); }}
              className="border border-dashed border-[#334155] bg-[#1E293B] p-6 text-center cursor-pointer hover:border-emerald-500" role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && input.current?.click()}>
              <Upload className="mx-auto mb-2 text-zinc-400" /><p className="text-sm">Kéo thả CV dạng PDF vào đây hoặc bấm để chọn (tối đa 10MB)</p>
              <input ref={input} type="file" accept="application/pdf" multiple hidden onChange={e => addFiles(e.target.files)} />
            </div>
            <textarea value={paste} onChange={e => setPaste(e.target.value)} rows={3} placeholder="Hoặc dán nội dung CV" className="w-full mt-2 bg-[#1E293B] border border-[#334155] p-2 text-sm" />
            <button onClick={addPaste} className="mt-1 text-sm text-emerald-400 hover:underline">Thêm CV đã dán</button>
          </div>
        </section>

        <section className="bg-[#0F172A] p-5 min-w-0">
          {!cands.length ? (
            <div className="border border-[#334155] bg-[#1E293B] p-8 text-center">
              <FileText className="mx-auto mb-3 text-zinc-400" />
              <p className="font-semibold">Chưa có ứng viên nào</p>
              <p className="text-sm text-zinc-400 mb-4">Tải CV lên hoặc thử nhanh với ba hồ sơ mẫu.</p>
              <div className="flex flex-wrap justify-center gap-2">
                {SAMPLES.map(s => <button key={s.id} onClick={() => setCands(c => [...c, { ...s, id: crypto.randomUUID() }])} className="border border-[#334155] hover:border-emerald-500 px-3 py-1.5 text-sm">{s.label}</button>)}
              </div>
            </div>
          ) : (
            <div className="border border-[#334155] overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-[#1E293B] text-left text-zinc-400"><tr>
                  <th className="p-3 w-8">#</th><th className="p-3">Ứng viên</th><th className="p-3">Hướng nghề</th><th className="p-3">Điểm</th><th className="p-3 min-w-56">Chi tiết</th><th className="p-3 w-8" /></tr></thead>
                <tbody>
                  {ranked.map((c, i) => (
                    <tr key={c.id} className="border-t border-[#334155] hover:bg-[#1E293B]/60">
                      <td className={`p-3 ${num} text-zinc-400`}>{c.result ? i + 1 : '-'}</td>
                      <td className="p-3">
                        <button className="font-semibold text-left hover:underline disabled:no-underline" disabled={!c.result} onClick={() => setOpen(c)}>{c.result?.parsed.name || c.label}</button>
                        <div className="text-xs text-zinc-400">{c.result?.parsed.email || c.label}</div>
                        {c.error && <div className="mt-1 flex items-center gap-1 text-xs text-rose-400"><AlertCircle size={12} />{c.error}
                          <button onClick={() => screen([c])} className="underline ml-1">Thử lại</button></div>}
                      </td>
                      <td className="p-3">{c.result && <><span className="border border-[#334155] px-2 py-0.5">{c.result.classification.track}</span>
                        <div className={`text-xs text-zinc-400 mt-1 ${num}`}>{(c.result.classification.confidence * 100).toFixed(1)}% tin cậy</div></>}</td>
                      <td className={`p-3 text-2xl font-semibold ${num} ${c.result ? tone(c.result.match.hybrid_score) : ''}`}>{c.result?.match.hybrid_score.toFixed(1)}</td>
                      <td className="p-3">{c.result && <div className="flex gap-3"><Meter label="Ngữ nghĩa" value={c.result.match.semantic} /><Meter label="Kỹ năng" value={c.result.match.skill_overlap} /><Meter label="Hướng nghề" value={c.result.match.track_fit} /></div>}</td>
                      <td className="p-3"><button aria-label="Xoá ứng viên" onClick={() => setCands(x => x.filter(y => y.id !== c.id))}><X size={14} className="text-zinc-500 hover:text-rose-400" /></button></td>
                    </tr>))}
                </tbody>
              </table>
            </div>
          )}

          <div className="mt-6 border border-[#334155] bg-[#1E293B] p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <h2 className="font-semibold flex items-center gap-2"><Brain size={18} />Mô hình phân loại</h2>
              <button onClick={retrain} disabled={training} className="flex items-center gap-2 border border-[#334155] hover:border-emerald-500 px-3 py-1.5 text-sm disabled:opacity-50">
                <RefreshCw size={14} className={training ? 'animate-spin' : ''} />{training ? 'Đang huấn luyện lại' : 'Huấn luyện lại'}</button>
            </div>
            {meta ? <>
              <div className={`grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm ${num}`}>
                <div><div className="text-xs text-zinc-400 font-sans">Độ chính xác (test)</div>{(meta.accuracy * 100).toFixed(1)}%</div>
                <div><div className="text-xs text-zinc-400 font-sans">Từ vựng</div>{meta.vocab_size}</div>
                <div><div className="text-xs text-zinc-400 font-sans">Mẫu huấn luyện</div>{meta.train_size}</div>
                <div><div className="text-xs text-zinc-400 font-sans">Embedding</div><span className="text-xs">{info.embedding_backend}</span></div>
              </div>
              <p className="text-xs text-zinc-400 mt-3">Tệp: {Object.entries(info.artifacts).map(([k, v]) => `${k} (${((v as number) / 1024).toFixed(0)} KB)`).join(', ')}</p>
              <p className="text-xs text-amber-400 mt-1">{meta.note}</p>
            </> : <p className="text-sm text-zinc-400">Chưa kết nối được dịch vụ mô hình. Chờ vài giây rồi tải lại trang.</p>}
            <div className="mt-4 flex flex-col sm:flex-row gap-2">
              <textarea value={testText} onChange={e => setTestText(e.target.value)} rows={2} placeholder="Dán đoạn CV để thử phân loại" className="flex-1 bg-[#0F172A] border border-[#334155] p-2 text-sm" />
              <button onClick={runTest} disabled={testText.length < 20} className="border border-[#334155] hover:border-emerald-500 px-3 text-sm disabled:opacity-40">Chạy thử</button>
            </div>
            {testOut && <p className={`mt-2 text-sm ${num}`}>{testOut}</p>}
          </div>
        </section>
      </main>

      {open?.result && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center p-4 z-50" role="dialog" aria-modal="true" onClick={() => setOpen(null)}>
          <div className="bg-[#1E293B] border border-[#334155] max-w-3xl w-full max-h-[90vh] overflow-y-auto p-6" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-start mb-4">
              <div><h2 className="text-xl font-semibold">{open.result.parsed.name}</h2>
                <p className={`text-xs text-zinc-400 ${num}`}>{[open.result.parsed.email, open.result.parsed.phone, ...open.result.parsed.links].filter(Boolean).join('  |  ')}</p></div>
              <button aria-label="Đóng" onClick={() => setOpen(null)}><X /></button>
            </div>
            <div className={`text-4xl font-semibold ${num} ${tone(open.result.match.hybrid_score)}`}>{open.result.match.hybrid_score.toFixed(1)}<span className="text-sm text-zinc-400"> / 100</span></div>
            <h3 className="font-semibold mt-5 mb-2">Xác suất theo hướng nghề</h3>
            <div className="space-y-2">{Object.entries(open.result.classification.probabilities).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
              <div key={k} className="flex items-center gap-3 text-sm"><span className="w-44">{k}</span><div className="flex-1 h-1.5 bg-[#334155]"><div className="h-full bg-emerald-500" style={{ width: `${v * 100}%` }} /></div><span className={`w-14 text-right ${num}`}>{(v * 100).toFixed(1)}%</span></div>)}</div>
            <h3 className="font-semibold mt-5 mb-2">So khớp kỹ năng</h3>
            <div className="grid sm:grid-cols-2 gap-4 text-sm">
              <div><p className="text-emerald-400 mb-1">Đã có ({open.result.match.matched.length})</p><div className="flex flex-wrap gap-1">{open.result.match.matched.map(s => <span key={s} className={`border border-emerald-500/40 px-2 py-0.5 ${num} text-xs`}>{s}</span>)}</div></div>
              <div><p className="text-rose-400 mb-1">Còn thiếu ({open.result.match.missing.length})</p><div className="flex flex-wrap gap-1">{open.result.match.missing.length ? open.result.match.missing.map(s => <span key={s} className={`border border-rose-500/40 px-2 py-0.5 ${num} text-xs`}>{s}</span>) : <span className="text-zinc-400">Không thiếu kỹ năng nào.</span>}</div></div>
            </div>
            <h3 className="font-semibold mt-5 mb-2">Văn bản trích xuất</h3>
            <pre className={`bg-[#0F172A] border border-[#334155] p-3 text-xs whitespace-pre-wrap max-h-60 overflow-y-auto ${num}`}>{open.result.parsed.text}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
