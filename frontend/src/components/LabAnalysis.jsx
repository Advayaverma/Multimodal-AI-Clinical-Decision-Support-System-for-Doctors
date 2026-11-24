import React, { useEffect, useState } from 'react';
import { FlaskConical, ShieldCheck, Sparkles, RotateCcw, Check, CheckCheck, Loader2 } from 'lucide-react';
import { patientService } from '../services/api';

const initialForm = { test_name: 'Glucose', value: '', unit: 'mg/dL', test_date: new Date().toISOString().slice(0, 10) };
const statusStyle = {
  Normal: 'bg-emerald-50 text-emerald-800 border-emerald-200',
  'Potentially High': 'bg-amber-50 text-amber-800 border-amber-200',
  'Potentially Low': 'bg-sky-50 text-sky-800 border-sky-200',
};

const normalizeUnit = (testName, rawUnit) => {
  const t = testName.toLowerCase();
  const u = (rawUnit || '').trim().toLowerCase();
  if (t.includes('sodium') || t.includes('potassium')) {
    if (u.includes('mmol') || u.includes('meq')) return 'mEq/L';
  }
  if (t.includes('glucose') || t.includes('creatinine') || t.includes('urea') || t.includes('cholesterol') || t.includes('ldl') || t.includes('hdl') || t.includes('triglyceride')) {
    if (u.includes('mg')) return 'mg/dL';
  }
  if (t.includes('hemoglobin') || t.includes('haemoglobin')) {
    if (u.includes('g/')) return 'g/dL';
  }
  if (t.includes('a1c')) return '%';
  if (t.includes('wbc') || t.includes('platelet')) return '/uL';
  if (t.includes('tsh')) return 'mIU/L';
  return rawUnit || '';
};

const extractOcrLabs = (reports) => {
  const ocrLabs = [];
  const seen = new Set();

  for (const report of reports || []) {
    // 1. Parse from pre-extracted entities if available
    const entities = report.clinical_entities?.lab_values || [];
    for (const ent of entities) {
      const name = ent.name || ent.text;
      const unit = normalizeUnit(name, ent.unit);
      const val = String(ent.value);
      const key = `${name.toLowerCase()}:${val}`;
      if (!seen.has(key)) {
        seen.add(key);
        ocrLabs.push({
          test_name: name,
          value: val,
          unit: unit || (name.toLowerCase() === 'glucose' ? 'mg/dL' : ''),
          source: report.filename,
          date: report.uploaded_at?.slice(0, 10) || new Date().toISOString().slice(0, 10),
        });
      }
    }

    // 2. Direct regex scan on extracted_text as client-side fallback
    const text = report.extracted_text || '';
    if (text) {
      const patterns = [
        [/\b(?:glucose(?:\s+pp|\s+fasting|\s+f|\s+r)?|blood\s+sugar|fbs|rbs|ppbs)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mg\/d[lL]|mmol\/[lL])?/gi, 'Glucose', 'mg/dL'],
        [/\b(?:glycated\s+h[ae]moglobin|glycosylated\s+h[ae]moglobin|hba1c|ha1c|a1c)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(%)?/gi, 'HbA1c', '%'],
        [/\b(?:h[ae]moglobin|hgb|hb)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(g\/d[lL]|g\/[lL])?/gi, 'Hemoglobin', 'g/dL'],
        [/\b(?:white\s+blood\s+cells?|wbc|total\s+leucocyte\s+count|tlc)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(\/uL|cells\/uL)?/gi, 'WBC', '/uL'],
        [/\b(?:platelets?|plt|platelet\s+count)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(\/uL|cells\/uL)?/gi, 'Platelets', '/uL'],
        [/\b(?:serum\s+creatinine|creatinine|creat)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mg\/d[lL])?/gi, 'Creatinine', 'mg/dL'],
        [/\b(?:blood\s+urea\s+nitrogen|serum\s+urea|urea|bun)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mg\/d[lL])?/gi, 'Urea', 'mg/dL'],
        [/\b(?:serum\s+sodium|sodium|na\+?)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mEq\/L|mmol\/L)?/gi, 'Sodium', 'mEq/L'],
        [/\b(?:serum\s+potassium|potassium|k\+?)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mEq\/L|mmol\/L)?/gi, 'Potassium', 'mEq/L'],
        [/\b(?:total\s+cholesterol|cholesterol)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mg\/d[lL])?/gi, 'Total Cholesterol', 'mg/dL'],
        [/\b(?:ldl(?:\s*-\s*c)?|ldl\s+cholesterol)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mg\/d[lL])?/gi, 'LDL', 'mg/dL'],
        [/\b(?:hdl(?:\s*-\s*c)?|hdl\s+cholesterol)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mg\/d[lL])?/gi, 'HDL', 'mg/dL'],
        [/\b(?:triglycerides?|tg)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mg\/d[lL])?/gi, 'Triglycerides', 'mg/dL'],
        [/\b(?:thyroid\s+stimulating\s+hormone|tsh)\b(?:\s*[:=-]?\s*|\n[^\n]*\n[^\n]*?)(\d+(?:\.\d+)?)\s*(mIU\/L|uIU\/mL)?/gi, 'TSH', 'mIU/L'],
      ];

      for (const [regex, canonical, defUnit] of patterns) {
        let m;
        while ((m = regex.exec(text)) !== null) {
          const val = m[1];
          const unit = normalizeUnit(canonical, m[2] || defUnit);
          const key = `${canonical.toLowerCase()}:${val}`;
          if (!seen.has(key)) {
            seen.add(key);
            ocrLabs.push({
              test_name: canonical,
              value: val,
              unit,
              source: report.filename,
              date: report.uploaded_at?.slice(0, 10) || new Date().toISOString().slice(0, 10),
            });
          }
        }
      }
    }
  }
  return ocrLabs;
};

export default function LabAnalysis({ patientId, refreshKey = 0 }) {
  const [results, setResults] = useState([]);
  const [form, setForm] = useState(initialForm);
  const [saving, setSaving] = useState(false);
  const [checkingAll, setCheckingAll] = useState(false);
  const [checkingItem, setCheckingItem] = useState(null);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [ocrLabs, setOcrLabs] = useState([]);
  const [autoFilled, setAutoFilled] = useState(null);

  const load = async () => {
    try {
      const [resLabs, resReports] = await Promise.all([
        patientService.listLabResults(patientId),
        patientService.listReports(patientId).catch(() => ({ data: [] })),
      ]);
      setResults(resLabs.data || []);

      const detected = extractOcrLabs(resReports.data || []);
      setOcrLabs(detected);

      // Auto-fill first lab finding if form value is currently empty
      if (detected.length > 0) {
        const glucoseItem = detected.find((item) => item.test_name.toLowerCase() === 'glucose') || detected[0];
        setForm((prev) => {
          if (!prev.value || prev.value === '') {
            return {
              ...prev,
              test_name: glucoseItem.test_name,
              value: glucoseItem.value,
              unit: glucoseItem.unit || 'mg/dL',
              test_date: glucoseItem.date,
            };
          }
          return prev;
        });
        setAutoFilled(glucoseItem);
      }
    } catch {
      setError('Unable to load laboratory results.');
    }
  };

  useEffect(() => {
    load();
  }, [patientId, refreshKey]);

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    setSuccessMsg('');
    setSaving(true);
    try {
      await patientService.addLabResult(patientId, {
        ...form,
        value: Number(form.value),
        unit: form.unit || undefined,
      });
      setForm(initialForm);
      setAutoFilled(null);
      await load();
    } catch (err) {
      setError(err.response?.data?.detail || 'Unable to analyse and save this result.');
    } finally {
      setSaving(false);
    }
  };

  const change = (event) => {
    setForm({ ...form, [event.target.name]: event.target.value });
    if (event.target.name === 'value' && autoFilled && event.target.value !== autoFilled.value) {
      setAutoFilled(null);
    }
  };

  const fillFromOcr = (item) => {
    setForm({
      test_name: item.test_name,
      value: item.value,
      unit: item.unit || '',
      test_date: item.date,
    });
    setAutoFilled(item);
  };

  // One-click input and evaluate a single extracted finding
  const checkSingleFinding = async (item) => {
    setError('');
    setSuccessMsg('');
    setCheckingItem(`${item.test_name}-${item.value}`);
    fillFromOcr(item);
    try {
      await patientService.addLabResult(patientId, {
        test_name: item.test_name,
        value: Number(item.value),
        unit: item.unit || undefined,
        test_date: item.date,
      });
      setSuccessMsg(`Checked & recorded ${item.test_name}: ${item.value} ${item.unit}`);
      await load();
    } catch (err) {
      setError(err.response?.data?.detail || `Failed to check ${item.test_name}.`);
    } finally {
      setCheckingItem(null);
    }
  };

  // Batch import and evaluate all findings extracted from reports
  const checkAllFindings = async () => {
    if (!ocrLabs.length) return;
    setError('');
    setSuccessMsg('');
    setCheckingAll(true);
    let successCount = 0;
    let lastErr = null;

    for (const item of ocrLabs) {
      try {
        await patientService.addLabResult(patientId, {
          test_name: item.test_name,
          value: Number(item.value),
          unit: item.unit || undefined,
          test_date: item.date,
        });
        successCount++;
      } catch (err) {
        lastErr = err.response?.data?.detail || err.message;
      }
    }

    setCheckingAll(false);
    if (successCount > 0) {
      setSuccessMsg(`Successfully checked and recorded ${successCount} lab finding(s) from report!`);
      await load();
    } else if (lastErr) {
      setError(`Failed to check findings: ${lastErr}`);
    }
  };

  const clearAutoFill = () => {
    setForm(initialForm);
    setAutoFilled(null);
  };

  // Determine if finding already checked in saved results
  const isFindingChecked = (item) => {
    return results.some(
      (r) =>
        r.test_name.toLowerCase() === item.test_name.toLowerCase() &&
        Math.abs(Number(r.value) - Number(item.value)) < 0.001
    );
  };

  return (
    <section className="med-card p-5 space-y-4">
      <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-[#0e7c5b]" />
          <div>
            <h2 className="text-sm font-bold text-slate-800">Laboratory Analysis & Reference Rules</h2>
            <p className="text-[11px] text-slate-500">
              Deterministic rule-based interpretation of patient biomarker and chemistry findings.
            </p>
          </div>
        </div>
        <span className="text-[10px] bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-0.5 rounded font-medium">
          Live Rule Engine
        </span>
      </div>

      {/* Auto-Extract from Reports Section */}
      {ocrLabs.length > 0 && (
        <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/50 shadow-2xs">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-emerald-200/60 pb-2.5">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-[#0e7c5b]" />
              <span className="text-xs font-bold text-emerald-950 uppercase tracking-wide">
                Findings Extracted from Reports ({ocrLabs.length})
              </span>
            </div>
            <button
              type="button"
              onClick={checkAllFindings}
              disabled={checkingAll}
              className="btn-med-primary text-xs py-1.5 px-3 flex items-center gap-1.5 shadow-xs w-full sm:w-auto justify-center"
            >
              {checkingAll ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Checking all findings…</span>
                </>
              ) : (
                <>
                  <CheckCheck className="w-3.5 h-3.5" />
                  <span>Check All Findings ({ocrLabs.length})</span>
                </>
              )}
            </button>
          </div>

          <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
            {ocrLabs.map((lab, idx) => {
              const checked = isFindingChecked(lab);
              const isCheckingThis = checkingItem === `${lab.test_name}-${lab.value}`;

              return (
                <div
                  key={`${lab.test_name}-${lab.value}-${idx}`}
                  className={`p-2.5 rounded-lg border text-xs flex flex-col justify-between gap-2 transition-all ${
                    checked
                      ? 'bg-white/80 border-emerald-300 shadow-2xs'
                      : 'bg-white border-slate-200 hover:border-emerald-300'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between gap-1">
                      <span className="font-bold text-slate-800">{lab.test_name}</span>
                      {checked && (
                        <span className="text-[10px] text-emerald-700 bg-emerald-100/70 px-1.5 py-0.5 rounded flex items-center gap-0.5 font-medium">
                          <Check className="w-2.5 h-2.5" /> Checked
                        </span>
                      )}
                    </div>
                    <div className="mt-1 flex items-baseline gap-1 text-sm font-semibold font-mono text-[#0e7c5b]">
                      <span>{lab.value}</span>
                      <span className="text-xs font-normal text-slate-500">{lab.unit}</span>
                    </div>
                    {lab.source && (
                      <span className="text-[10px] text-slate-400 block truncate mt-0.5" title={lab.source}>
                        Source: {lab.source}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-1.5 pt-1 border-t border-slate-100">
                    <button
                      type="button"
                      disabled={isCheckingThis || checkingAll}
                      onClick={() => checkSingleFinding(lab)}
                      className="flex-1 text-[11px] py-1 px-2 rounded font-medium bg-[#0e7c5b] text-white hover:bg-[#0b6349] transition-colors flex items-center justify-center gap-1"
                    >
                      {isCheckingThis ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <Check className="w-3 h-3" />
                      )}
                      <span>Input & Check</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => fillFromOcr(lab)}
                      className="text-[11px] py-1 px-2 rounded font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 transition-colors"
                      title="Load into manual form below"
                    >
                      Fill Form
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Auto-filled Notification Banner */}
      {autoFilled && form.value === autoFilled.value && (
        <div className="flex items-center justify-between gap-2 p-2.5 rounded-lg border border-emerald-200 bg-emerald-50/70 text-xs text-emerald-900 animate-in fade-in">
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>
              Loaded <strong>{autoFilled.test_name}</strong> into form: <strong className="font-mono">{autoFilled.value} {autoFilled.unit}</strong>
              {autoFilled.source && <span className="text-emerald-700 ml-1">({autoFilled.source})</span>}
            </span>
          </div>
          <button
            type="button"
            onClick={clearAutoFill}
            className="flex items-center gap-1 text-[11px] text-emerald-700 hover:text-emerald-950 font-medium underline px-1 py-0.5"
            title="Reset form"
          >
            <RotateCcw className="w-3 h-3" />
            Clear
          </button>
        </div>
      )}

      {/* Manual Input Form */}
      <div>
        <p className="text-xs text-slate-600 mb-2">
          Or manually enter/adjust a lab value to run through the reference rules:
        </p>

        <form onSubmit={submit} className="grid grid-cols-1 sm:grid-cols-4 gap-2">
          <input
            name="test_name"
            value={form.test_name}
            onChange={change}
            className="input-med text-xs"
            placeholder="Test (e.g. Glucose)"
            required
          />
          <input
            name="value"
            type="number"
            step="any"
            value={form.value}
            onChange={change}
            className="input-med text-xs font-mono font-medium"
            placeholder="Result (e.g. 140)"
            required
          />
          <input
            name="unit"
            value={form.unit}
            onChange={change}
            className="input-med text-xs"
            placeholder="Unit (mg/dL)"
          />
          <div className="flex gap-2">
            <input
              name="test_date"
              type="date"
              value={form.test_date}
              onChange={change}
              className="input-med text-xs min-w-0"
            />
            <button
              disabled={saving || !form.value}
              className="btn-med-primary text-xs px-3 whitespace-nowrap"
            >
              {saving ? 'Checking…' : 'Analyse'}
            </button>
          </div>
        </form>

        <p className="mt-1.5 text-[10px] text-slate-400">
          Supported: Glucose, HbA1c, Hemoglobin, WBC, Platelets, Creatinine, Urea, Sodium, Potassium, Cholesterol, LDL, HDL, Triglycerides, TSH.
        </p>
      </div>

      {error && <p className="text-xs text-rose-700 bg-rose-50 border border-rose-200 p-2 rounded-lg">{error}</p>}
      {successMsg && <p className="text-xs text-emerald-800 bg-emerald-50 border border-emerald-200 p-2 rounded-lg">{successMsg}</p>}

      {/* Results Section */}
      <div className="pt-2 border-t border-slate-100">
        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wide mb-2">
          Evaluated Lab Results ({results.length})
        </h3>
        <div className="space-y-2">
          {!results.length ? (
            <p className="text-xs text-slate-400">No clinician-reviewed laboratory results recorded yet.</p>
          ) : (
            results.map((result) => (
              <div key={result.id} className="rounded-lg border border-slate-100 p-3 text-xs bg-white shadow-2xs">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-slate-800">
                    {result.test_name}: <span className="font-mono">{result.value}</span> {result.unit}
                  </span>
                  <span className={`border rounded-full px-2.5 py-0.5 font-semibold text-[11px] ${statusStyle[result.status] || 'bg-slate-50 text-slate-700 border-slate-200'}`}>
                    {result.status}
                  </span>
                </div>
                <p className="mt-1 text-slate-500">
                  Reference: {result.reference_range}. {result.clinical_basis}
                </p>
                <p className="mt-1 flex items-center gap-1 text-[11px] text-slate-400">
                  <ShieldCheck className="w-3 h-3 text-slate-400" /> Doctor review required
                  {result.test_date ? ` · Test date: ${result.test_date}` : ''}
                </p>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
