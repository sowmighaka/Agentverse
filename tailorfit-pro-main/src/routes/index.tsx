import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState, type FormEvent } from "react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8002";

export const Route = createFileRoute("/")({
  component: TailorFit,
});

type JobInputType = "text" | "url";
type TabKey = "resume" | "cover" | "changes" | "sources" | "steps" | "validation";

type ResumeProject = {
  title?: string;
  description?: string;
  tech_used?: string[];
  metrics?: string[];
};

type ResumeExperience = {
  role?: string;
  company?: string;
  dates?: string;
  bullets?: string[];
};

type ResumeDict = {
  name?: string;
  contact?: string;
  summary?: string;
  skills?: string[];
  experience?: ResumeExperience[];
  projects?: ResumeProject[];
  education?: unknown[];
  [key: string]: unknown;
};

type SourceChecked = {
  source_type: "job_posting" | "blog" | "github";
  identifier: string;
  status: "fetched" | "skipped" | "failed";
  summary: string;
};

type AgentTrace = {
  node_name: string;
  trace_note: string;
};

type KeywordAnalysis = {
  total_keywords_from_job: number;
  keywords_integrated: number;
  integration_rate: number;
  missing_critical_keywords: string[];
  naturally_integrated_keywords: string[];
  forced_keywords: string[];
};

type PhraseAnalysis = {
  phrases_used_correctly: string[];
  phrases_used_incorrectly: string[];
  missing_important_phrases: string[];
};

type ValidationFeedback = {
  strengths: string[];
  weaknesses: string[];
  suggestions: string[];
  ats_score: number;
  human_readability_score: number;
};

type ValidationReport = {
  passed_validation: boolean;
  overall_score: number;
  keyword_analysis: KeywordAnalysis;
  phrase_analysis: PhraseAnalysis;
  feedback: ValidationFeedback;
  ready_for_generation: boolean;
};

type TailorResponse = {
  session_id: string;
  rewritten_resume: ResumeDict;
  cover_letter: string;
  diff_explanation: string;
  verification_status: "passed" | "failed_after_retries";
  confidence_notes: string[];
  sources_checked: SourceChecked[];
  agent_trace: AgentTrace[];
  validation_report?: ValidationReport;
};

type StoredSession = {
  response: TailorResponse;
  resume: ResumeDict;
  coverLetter: string;
};

const STATUS_STEPS = [
  "Reading job posting...",
  "Analyzing engineering blog...",
  "Checking GitHub tech stack...",
  "Matching your experience...",
  "Rewriting your resume...",
  "Fact-checking the result...",
  "Formatting resume preview...",
];

function sourceStatusIcon(status: SourceChecked["status"]) {
  if (status === "fetched") return "OK";
  if (status === "skipped") return "SKIP";
  return "FAIL";
}
function storageKey(sessionId: string) {
  return `tailorfit:${sessionId}`;
}

function splitDiffBullets(markdown: string): string[] {
  return markdown
    .split("\n")
    .map((line) => line.trim().replace(/^[-*]\s*/, ""))
    .filter(Boolean);
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function asText(value: unknown): string {
  if (value == null) return "";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    if (obj.institution || obj.school || obj.degree || obj.university || obj.college || obj.academy) {
      const inst = (obj.institution || obj.school || obj.university || obj.college || obj.academy || "") as string;
      const deg = (obj.degree || "") as string;
      const dates = (obj.dates || obj.date || obj.years || obj.graduation_year || "") as string;
      const gpa = (obj.gpa || obj.score || obj.percentage || obj.grade || "") as string;
      const parts = [deg, inst, dates].filter(Boolean);
      const main = parts.join(", ");
      return gpa ? `${main} (GPA: ${gpa})` : main;
    }
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

function parseLines(value: string) {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function useReveal<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add("tf-in");
            io.unobserve(e.target);
          }
        }
      },
      { threshold: 0.15 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return ref;
}

function Reveal({
  children,
  className = "",
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  const ref = useReveal<HTMLDivElement>();
  return (
    <div ref={ref} className={`tf-reveal ${className}`} style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

function StatusRotator() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setI((v) => (v + 1) % STATUS_STEPS.length), 2000);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="flex items-center gap-3">
      <span className="flex items-center gap-1.5">
        <span className="tf-dot" />
        <span className="tf-dot" />
        <span className="tf-dot" />
      </span>
      <span key={i} className="tf-fade text-sm text-[#a1a1aa] font-mono">
        {STATUS_STEPS[i]}
      </span>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mt-6 border-b border-[#1A1A1A] pb-1 text-xs font-bold uppercase tracking-wider text-[#1A1A1A]">
      {children}
    </h3>
  );
}

function EditableText({
  value,
  onChange,
  multiline = false,
  placeholder = "",
}: {
  value: string;
  onChange: (value: string) => void;
  multiline?: boolean;
  placeholder?: string;
}) {
  if (multiline) {
    return (
      <textarea
        className="tf-input min-h-[92px] resize-y text-sm"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }
  return (
    <input
      className="tf-input text-sm"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function ResumePreview({
  resume,
  editing,
  onChange,
}: {
  resume: ResumeDict;
  editing: boolean;
  onChange: (resume: ResumeDict) => void;
}) {
  const experience = asArray<ResumeExperience>(resume.experience);
  const projects = asArray<ResumeProject>(resume.projects);
  const education = asArray<unknown>(resume.education);
  const skills = asArray<string>(resume.skills);

  function patch(next: Partial<ResumeDict>) {
    onChange({ ...resume, ...next });
  }

  function patchExperience(index: number, next: Partial<ResumeExperience>) {
    const updated = experience.map((item, i) => (i === index ? { ...item, ...next } : item));
    patch({ experience: updated });
  }

  function patchProject(index: number, next: Partial<ResumeProject>) {
    const updated = projects.map((item, i) => (i === index ? { ...item, ...next } : item));
    patch({ projects: updated });
  }

  return (
    <div className="rounded-md bg-white p-8 text-[#18181b] shadow-2xl">
      <div className="text-center">
        {editing ? (
          <div className="space-y-3">
            <EditableText
              value={asText(resume.name)}
              onChange={(v) => patch({ name: v })}
              placeholder="Name"
            />
            <EditableText
              value={asText(resume.contact)}
              onChange={(v) => patch({ contact: v })}
              placeholder="Contact"
            />
          </div>
        ) : (
          <>
            <h2 className="text-2xl font-bold tracking-wide">
              {asText(resume.name) || "Tailored Resume"}
            </h2>
            {resume.contact && (
              <p className="mt-1 text-sm text-[#52525b]">{asText(resume.contact)}</p>
            )}
          </>
        )}
      </div>

      {(editing || resume.summary) && (
        <>
          <SectionTitle>Summary</SectionTitle>
          {editing ? (
            <EditableText
              multiline
              value={asText(resume.summary)}
              onChange={(v) => patch({ summary: v })}
            />
          ) : (
            <p className="mt-2 text-sm leading-relaxed">{asText(resume.summary)}</p>
          )}
        </>
      )}

      {(editing || skills.length > 0) && (
        <>
          <SectionTitle>Skills</SectionTitle>
          {editing ? (
            <EditableText
              value={skills.join(", ")}
              onChange={(v) =>
                patch({
                  skills: v
                    .split(",")
                    .map((s) => s.trim())
                    .filter(Boolean),
                })
              }
            />
          ) : (
            <p className="mt-2 text-sm leading-relaxed">{skills.join(", ")}</p>
          )}
        </>
      )}

      {(editing || experience.length > 0) && (
        <>
          <SectionTitle>Experience</SectionTitle>
          <div className="mt-2 space-y-4">
            {experience.map((item, index) => (
              <div key={index}>
                {editing ? (
                  <div className="space-y-2">
                    <EditableText
                      value={asText(item.role)}
                      onChange={(v) => patchExperience(index, { role: v })}
                      placeholder="Role"
                    />
                    <EditableText
                      value={asText(item.company)}
                      onChange={(v) => patchExperience(index, { company: v })}
                      placeholder="Company"
                    />
                    <EditableText
                      value={asText(item.dates)}
                      onChange={(v) => patchExperience(index, { dates: v })}
                      placeholder="Dates"
                    />
                    <EditableText
                      multiline
                      value={asArray<string>(item.bullets).join("\n")}
                      onChange={(v) => patchExperience(index, { bullets: parseLines(v) })}
                      placeholder="One bullet per line"
                    />
                  </div>
                ) : (
                  <>
                    <p className="text-sm font-semibold">
                      {[item.role, item.company, item.dates].filter(Boolean).join(" - ")}
                    </p>
                    <ul className="mt-1 list-disc space-y-1 pl-5 text-sm leading-relaxed">
                      {asArray<string>(item.bullets).map((bullet, i) => (
                        <li key={i}>{bullet}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {(editing || projects.length > 0) && (
        <>
          <SectionTitle>Projects</SectionTitle>
          <div className="mt-2 space-y-4">
            {projects.map((item, index) => (
              <div key={index}>
                {editing ? (
                  <div className="space-y-2">
                    <EditableText
                      value={asText(item.title)}
                      onChange={(v) => patchProject(index, { title: v })}
                      placeholder="Project title"
                    />
                    <EditableText
                      multiline
                      value={asText(item.description)}
                      onChange={(v) => patchProject(index, { description: v })}
                      placeholder="Description"
                    />
                    <EditableText
                      value={asArray<string>(item.tech_used).join(", ")}
                      onChange={(v) =>
                        patchProject(index, {
                          tech_used: v
                            .split(",")
                            .map((s) => s.trim())
                            .filter(Boolean),
                        })
                      }
                      placeholder="Tech stack"
                    />
                    <EditableText
                      multiline
                      value={asArray<string>(item.metrics).join("\n")}
                      onChange={(v) => patchProject(index, { metrics: parseLines(v) })}
                      placeholder="One metric/bullet per line"
                    />
                  </div>
                ) : (
                  <>
                    <p className="text-sm font-semibold">
                      {item.title}
                      {asArray<string>(item.tech_used).length
                        ? ` (${asArray<string>(item.tech_used).join(", ")})`
                        : ""}
                    </p>
                    {item.description && (
                      <p className="mt-1 text-sm leading-relaxed">{item.description}</p>
                    )}
                    {asArray<string>(item.metrics).length > 0 && (
                      <ul className="mt-1 list-disc space-y-1 pl-5 text-sm leading-relaxed">
                        {asArray<string>(item.metrics).map((metric, i) => (
                          <li key={i}>{metric}</li>
                        ))}
                      </ul>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {(editing || education.length > 0) && (
        <>
          <SectionTitle>Education</SectionTitle>
          {editing ? (
            <EditableText
              multiline
              value={education.map(asText).join("\n")}
              onChange={(v) => patch({ education: parseLines(v) })}
            />
          ) : (
            <div className="mt-2 space-y-1 text-sm leading-relaxed">
              {education.map((item, index) => (
                <p key={index}>{asText(item)}</p>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function TailorFit() {
  const [resume, setResume] = useState("");
  const [jobInputType, setJobInputType] = useState<JobInputType>("text");
  const [jobInput, setJobInput] = useState("");
  const [blogUrls, setBlogUrls] = useState("");
  const [githubOrg, setGithubOrg] = useState("");
  const [loading, setLoading] = useState(false);
  const [resumeUploading, setResumeUploading] = useState(false);
  const [resumeFileName, setResumeFileName] = useState("");
  const [pdfLoading, setPdfLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TailorResponse | null>(null);
  const [editedResume, setEditedResume] = useState<ResumeDict | null>(null);
  const [editedCover, setEditedCover] = useState("");
  const [editing, setEditing] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>("resume");
  const [copied, setCopied] = useState<string | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const session = new URLSearchParams(window.location.search).get("session");
    if (!session) return;
    const raw = localStorage.getItem(storageKey(session));
    if (!raw) return;
    try {
      const stored = JSON.parse(raw) as StoredSession;
      setResult(stored.response);
      setEditedResume(stored.resume);
      setEditedCover(stored.coverLetter);
    } catch {
      localStorage.removeItem(storageKey(session));
    }
  }, []);

  useEffect(() => {
    if (result && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [result]);

  useEffect(() => {
    if (!result || !editedResume) return;
    const timer = window.setTimeout(() => {
      const stored: StoredSession = {
        response: result,
        resume: editedResume,
        coverLetter: editedCover,
      };
      localStorage.setItem(storageKey(result.session_id), JSON.stringify(stored));
    }, 500);
    return () => window.clearTimeout(timer);
  }, [result, editedResume, editedCover]);

  async function onResumeFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setResumeUploading(true);
    setResumeFileName(file.name);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${BACKEND_URL}/api/extract-resume-text`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `Resume upload failed (${res.status})`);
      }
      const data = (await res.json()) as { resume_text: string };
      setResume(data.resume_text);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not read the resume file.");
      setResumeFileName("");
    } finally {
      setResumeUploading(false);
      e.target.value = "";
    }
  }
  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/tailor`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume_text: resume,
          job_input: jobInput,
          job_input_type: jobInputType,
          blog_urls: blogUrls
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          github_org: githubOrg.trim() || null,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `Request failed (${res.status})`);
      }
      const data = (await res.json()) as TailorResponse;
      setResult(data);
      setEditedResume(data.rewritten_resume);
      setEditedCover(data.cover_letter);
      setActiveTab("resume");
      const url = new URL(window.location.href);
      url.searchParams.set("session", data.session_id);
      window.history.replaceState({}, "", url);
      localStorage.setItem(
        storageKey(data.session_id),
        JSON.stringify({
          response: data,
          resume: data.rewritten_resume,
          coverLetter: data.cover_letter,
        } satisfies StoredSession),
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Something went wrong reaching the tailoring service.",
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setResult(null);
    setEditedResume(null);
    setEditedCover("");
    setError(null);
    setEditing(false);
    const url = new URL(window.location.href);
    url.searchParams.delete("session");
    window.history.replaceState({}, "", url);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function copy(text: string, label: string) {
    await navigator.clipboard.writeText(text);
    setCopied(label);
    setTimeout(() => setCopied(null), 1500);
  }

  async function downloadPdf() {
    if (!editedResume) return;
    setPdfLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/render-resume-pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ resume: editedResume }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `PDF request failed (${res.status})`);
      }
      const blob = await res.blob();
      const href = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = href;
      a.download = "tailored-resume.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(href);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not download PDF.");
    } finally {
      setPdfLoading(false);
    }
  }

  const spotRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = spotRef.current;
    if (!el) return;
    const onMove = (e: MouseEvent) => {
      el.style.left = `${e.clientX}px`;
      el.style.top = `${e.clientY}px`;
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  const tabs: { key: TabKey; label: string }[] = [
    { key: "resume", label: "Tailored Resume" },
    { key: "cover", label: "Cover Letter" },
    { key: "changes", label: "Why These Changes" },
    { key: "validation", label: "Scores & Feedback" },
    { key: "sources", label: "Sources Checked" },
    { key: "steps", label: "Agent Steps" },
  ];

  return (
    <div className="relative min-h-screen">
      <div className="tf-bg" aria-hidden>
        <div className="tf-aurora" />
        <div className="tf-glow-violet" />
        <div className="tf-glow-cyan" />
        <div className="tf-dots" />
        <div className="tf-scanline" />
      </div>
      <div ref={spotRef} className="tf-spotlight" aria-hidden />

      <main className="relative mx-auto max-w-6xl px-6 pb-32 pt-20 sm:pt-28">
        <section className="mx-auto max-w-3xl text-center">
          <span className="glass-chip tf-chip-glow inline-flex items-center px-3 py-1 text-xs">
            <span className="mr-2 inline-block h-1.5 w-1.5 rounded-full bg-[#34d399] shadow-[0_0_8px_#34d399]" />
            <span className="text-gradient font-medium">AI-powered application tailoring</span>
          </span>
          <h1 className="mt-6 text-4xl font-bold tracking-tight text-[#f5f5f7] sm:text-6xl">
            Tailored to what they <span className="tf-shimmer">actually build</span>
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-base text-[#a1a1aa] sm:text-lg">
            Paste your resume and a target role. The backend checks sources, rewrites carefully,
            verifies claims, and formats the result.
          </p>
        </section>

        <Reveal className="mx-auto mt-12 max-w-[760px]">
          <form
            onSubmit={onSubmit}
            className={`glass-panel p-6 sm:p-8 transition-opacity ${loading ? "pointer-events-none opacity-60" : ""}`}
          >
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-[#f5f5f7]">Your resume</span>
              <div className="mb-3 rounded-md border border-white/10 bg-white/5 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-[#f5f5f7]">Upload resume file</p>
                    <p className="mt-1 text-xs text-[#a1a1aa]">
                      Supports .txt, .pdf, and .docx. Extracted text will appear below.
                    </p>
                  </div>
                  <label className="btn-outline-gradient cursor-pointer px-4 py-2 text-sm">
                    {resumeUploading ? "Reading..." : "Choose file"}
                    <input
                      className="sr-only"
                      type="file"
                      accept=".txt,.pdf,.docx,text/plain,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                      onChange={onResumeFileChange}
                      disabled={resumeUploading || loading}
                    />
                  </label>
                </div>
                {resumeFileName && (
                  <p className="mt-3 text-xs font-mono text-[#34d399]">Loaded: {resumeFileName}</p>
                )}
              </div>
              <textarea
                className="tf-input min-h-[180px] resize-y"
                placeholder="Paste your resume text here or upload a file above"
                value={resume}
                onChange={(e) => setResume(e.target.value)}
                required
              />
            </label>

            <div className="mt-6">
              <span className="mb-2 block text-sm font-medium text-[#f5f5f7]">Target company</span>
              <div className="glass-chip relative mb-3 inline-flex p-1 text-xs">
                {(["text", "url"] as const).map((v) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setJobInputType(v)}
                    className={`relative z-10 rounded-full px-4 py-1.5 font-medium transition-colors ${jobInputType === v ? "text-white" : "text-[#a1a1aa]"}`}
                  >
                    {jobInputType === v && (
                      <span className="absolute inset-0 -z-10 rounded-full bg-gradient-to-r from-[#8b5cf6] to-[#d946ef]" />
                    )}
                    {v === "text" ? "Paste job description" : "Job posting URL"}
                  </button>
                ))}
              </div>
              {jobInputType === "text" ? (
                <textarea
                  className="tf-input min-h-[120px] resize-y"
                  placeholder="Paste the full job description"
                  value={jobInput}
                  onChange={(e) => setJobInput(e.target.value)}
                  required
                />
              ) : (
                <input
                  className="tf-input"
                  type="url"
                  placeholder="https://company.com/careers/senior-engineer"
                  value={jobInput}
                  onChange={(e) => setJobInput(e.target.value)}
                  required
                />
              )}
            </div>

            <label className="mt-5 block">
              <span className="mb-2 block text-sm font-medium text-[#f5f5f7]">
                Engineering blog URL(s){" "}
                <span className="text-[#71717a]">optional, comma separated</span>
              </span>
              <input
                className="tf-input"
                placeholder="https://company.com/blog, https://eng.company.com"
                value={blogUrls}
                onChange={(e) => setBlogUrls(e.target.value)}
              />
            </label>

            <label className="mt-5 block">
              <span className="mb-2 block text-sm font-medium text-[#f5f5f7]">
                GitHub org name <span className="text-[#71717a]">optional</span>
              </span>
              <input
                className="tf-input"
                placeholder="e.g. vercel"
                value={githubOrg}
                onChange={(e) => setGithubOrg(e.target.value)}
              />
            </label>

            <button
              type="submit"
              className="btn-gradient tf-sheen mt-8 w-full px-6 py-3.5 text-base"
              disabled={loading}
            >
              {loading ? "Working..." : "Generate Tailored Application"}
            </button>
            {loading && (
              <div className="mt-6 flex justify-center">
                <StatusRotator />
              </div>
            )}
          </form>

          {error && (
            <div className="glass-panel mt-6 border-l-4 border-l-[#ef4444] p-5">
              <p className="text-sm font-medium text-[#f5f5f7]">
                We couldn't reach the tailoring service.
              </p>
              <p className="mt-1 text-sm text-[#a1a1aa]">{error}</p>
            </div>
          )}
        </Reveal>

        {result && editedResume && (
          <section ref={resultsRef} className="mt-20 scroll-mt-16">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-2xl font-bold tracking-tight text-[#f5f5f7]">
                  Your <span className="text-gradient">tailored</span> application
                </h2>
                <p className="mt-1 text-xs font-mono text-[#a1a1aa]">Session {result.session_id}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setEditing((v) => !v)}
                  className="btn-outline-gradient px-4 py-2 text-sm"
                >
                  {editing ? "Preview" : "Edit"}
                </button>
                <button
                  type="button"
                  onClick={downloadPdf}
                  className="btn-gradient px-4 py-2 text-sm"
                  disabled={pdfLoading}
                >
                  {pdfLoading ? "Preparing PDF..." : "Download PDF"}
                </button>
                <button
                  type="button"
                  onClick={reset}
                  className="btn-outline-gradient px-4 py-2 text-sm"
                >
                  Start over
                </button>
              </div>
            </div>

            <div
              className="glass-panel border-l-4 p-5"
              style={{
                borderLeftColor: result.verification_status === "passed" ? "#34d399" : "#fbbf24",
              }}
            >
              <p className="text-sm font-medium text-[#f5f5f7]">
                {result.verification_status === "passed"
                  ? "Fact-checked: no unsupported claims found"
                  : "Verifier could not fully pass this after retries"}
              </p>
              {result.confidence_notes.length > 0 && (
                <p className="mt-1 text-sm text-[#a1a1aa]">{result.confidence_notes.join(" | ")}</p>
              )}
            </div>

            <div className="mt-6 flex flex-wrap gap-2">
              {tabs.map((tab) => (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => setActiveTab(tab.key)}
                  className={`rounded-md border px-3 py-2 text-sm transition ${activeTab === tab.key ? "border-[#d946ef] bg-[#d946ef]/20 text-white" : "border-white/10 bg-white/5 text-[#a1a1aa]"}`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="mt-6">
              {activeTab === "resume" && (
                <ResumePreview resume={editedResume} editing={editing} onChange={setEditedResume} />
              )}

              {activeTab === "cover" && (
                <div className="glass-panel p-6">
                  <div className="mb-3 flex items-center justify-between">
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-[#f5f5f7]">
                      Cover Letter
                    </h3>
                    <button
                      type="button"
                      onClick={() => copy(editedCover, "cover")}
                      className="btn-outline-gradient px-3 py-1.5 text-xs"
                    >
                      {copied === "cover" ? "Copied" : "Copy"}
                    </button>
                  </div>
                  {editing ? (
                    <textarea
                      className="tf-input min-h-[220px] resize-y"
                      value={editedCover}
                      onChange={(e) => setEditedCover(e.target.value)}
                    />
                  ) : (
                    <p className="whitespace-pre-wrap text-sm leading-relaxed text-[#f5f5f7]">
                      {editedCover}
                    </p>
                  )}
                </div>
              )}

              {activeTab === "changes" && (
                <div className="glass-panel p-6">
                  <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[#f5f5f7]">
                    Why These Changes
                  </h3>
                  <ul className="space-y-2 text-sm text-[#a1a1aa]">
                    {splitDiffBullets(result.diff_explanation).map((item, index) => (
                      <li key={index} className="flex gap-2">
                        <span className="text-gradient">-&gt;</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {activeTab === "validation" && result.validation_report && (
                <div className="glass-panel p-6 space-y-6">
                  <div>
                    <h3 className="text-lg font-bold text-[#f5f5f7]">ATS Optimization & QA Feedback</h3>
                    <p className="text-xs text-[#a1a1aa] mt-1">Weighted score breakdown: 40% Keyword, 30% ATS, 30% Readability</p>
                  </div>
                  
                  {/* Scores Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="rounded-md border border-white/10 bg-white/5 p-4 text-center">
                      <span className="text-xs text-[#a1a1aa] uppercase tracking-wider">Overall Score</span>
                      <div className="text-3xl font-extrabold text-[#34d399] mt-2">{result.validation_report.overall_score}/100</div>
                    </div>
                    <div className="rounded-md border border-white/10 bg-white/5 p-4 text-center">
                      <span className="text-xs text-[#a1a1aa] uppercase tracking-wider">ATS Score</span>
                      <div className="text-3xl font-extrabold text-[#d946ef] mt-2">{result.validation_report.feedback.ats_score}/100</div>
                    </div>
                    <div className="rounded-md border border-white/10 bg-white/5 p-4 text-center">
                      <span className="text-xs text-[#a1a1aa] uppercase tracking-wider">Readability Score</span>
                      <div className="text-3xl font-extrabold text-[#38bdf8] mt-2">{result.validation_report.feedback.human_readability_score}/100</div>
                    </div>
                  </div>

                  {/* Keyword analysis */}
                  <div className="rounded-md border border-white/10 bg-white/5 p-4 space-y-3">
                    <h4 className="text-sm font-semibold text-[#f5f5f7] uppercase tracking-wider">Keyword Analysis</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                      <div>
                        <span className="text-[#a1a1aa]">Total Keywords:</span> <span className="font-mono text-white">{result.validation_report.keyword_analysis.total_keywords_from_job}</span>
                      </div>
                      <div>
                        <span className="text-[#a1a1aa]">Integrated:</span> <span className="font-mono text-white">{result.validation_report.keyword_analysis.keywords_integrated}</span>
                      </div>
                      <div>
                        <span className="text-[#a1a1aa]">Integration Rate:</span> <span className="font-mono text-[#34d399]">{result.validation_report.keyword_analysis.integration_rate.toFixed(1)}%</span>
                      </div>
                    </div>
                    {result.validation_report.keyword_analysis.missing_critical_keywords.length > 0 && (
                      <div className="text-xs space-y-1">
                        <span className="text-[#fbbf24] font-medium font-semibold">Missing Critical Keywords:</span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {result.validation_report.keyword_analysis.missing_critical_keywords.map((kw, i) => (
                            <span key={i} className="px-2 py-0.5 rounded bg-[#fbbf24]/10 text-[#fbbf24] border border-[#fbbf24]/20">{kw}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Strengths / Weaknesses */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                    {/* Strengths */}
                    {result.validation_report.feedback.strengths.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="font-bold text-[#34d399] uppercase tracking-wider text-xs">✓ Strengths</h4>
                        <ul className="list-disc pl-5 space-y-1 text-[#a1a1aa]">
                          {result.validation_report.feedback.strengths.map((str, i) => <li key={i}>{str}</li>)}
                        </ul>
                      </div>
                    )}
                    
                    {/* Weaknesses */}
                    {result.validation_report.feedback.weaknesses.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="font-bold text-[#f87171] uppercase tracking-wider text-xs">✗ Weaknesses</h4>
                        <ul className="list-disc pl-5 space-y-1 text-[#a1a1aa]">
                          {result.validation_report.feedback.weaknesses.map((weak, i) => <li key={i}>{weak}</li>)}
                        </ul>
                      </div>
                    )}
                  </div>
                  
                  {/* Suggestions */}
                  {result.validation_report.feedback.suggestions.length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-white/5">
                      <h4 className="font-bold text-[#38bdf8] uppercase tracking-wider text-xs">→ Actionable Suggestions</h4>
                      <ul className="list-disc pl-5 space-y-1 text-[#a1a1aa] text-sm">
                        {result.validation_report.feedback.suggestions.map((sug, i) => <li key={i}>{sug}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {activeTab === "sources" && (
                <div className="glass-panel p-6">
                  <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-[#f5f5f7]">
                    Sources Checked
                  </h3>
                  <div className="space-y-3">
                    {result.sources_checked.map((source, index) => (
                      <div
                        key={index}
                        className={`rounded-md border p-4 ${source.status === "failed" ? "border-[#ef4444]/60 bg-[#ef4444]/10" : source.status === "skipped" ? "border-[#fbbf24]/50 bg-[#fbbf24]/10" : "border-[#34d399]/40 bg-[#34d399]/10"}`}
                      >
                        <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-[#f5f5f7]">
                          <span className="rounded border border-white/20 px-1.5 py-0.5 font-mono text-[11px]">
                            {sourceStatusIcon(source.status)}
                          </span>
                          <span className="font-mono uppercase">{source.status}</span>
                          <span>{source.source_type}</span>
                          <span className="break-all text-[#a1a1aa]">{source.identifier}</span>
                        </div>
                        <p className="mt-1 text-sm text-[#d4d4d8]">{source.summary}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === "steps" && (
                <div className="glass-panel p-6">
                  <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-[#f5f5f7]">
                    Agent Steps
                  </h3>
                  <ol className="space-y-3 text-sm text-[#a1a1aa]">
                    {result.agent_trace.map((trace, index) => (
                      <li key={`${trace.node_name}-${index}`} className="flex gap-3">
                        <span className="font-mono text-[#d946ef]">{index + 1}.</span>
                        <span>
                          <strong className="text-[#f5f5f7]">{trace.node_name}</strong>:{" "}
                          {trace.trace_note}
                        </span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
