/**
 * Job polling.
 *
 * Module D's fit does not finish inside a request, so the backend accepts the work and
 * hands back a handle. This is the client half of that handshake: poll until the job is
 * terminal, and stop cleanly when the component unmounts.
 *
 * **Why a backoff rather than a fixed interval.** A detection run takes seconds to
 * minutes. Polling every 250 ms for two minutes is 480 requests, each one a SQLite read
 * plus a JSON serialisation, all to learn nothing has changed. The interval starts short
 * so a fast run reports immediately and climbs to a ceiling so a slow one costs a handful
 * of requests a minute instead of hundreds.
 *
 * **Why it does not give up on its own.** There is no sensible timeout: a 50 000-bar
 * series legitimately takes a while, and a client-side deadline that fired first would
 * report a failure the backend never had. The caller owns cancellation (`signal`), which
 * is what unmounting does. An abandoned run keeps going server-side and still lands in the
 * job registry, so nothing is lost by walking away.
 */

import { api } from "./client";
import type { JobRecord, RegimeRunResult } from "./types";

/** Interval between polls, in ms, indexed by attempt and clamped to the last entry. */
const BACKOFF_MS = [250, 400, 650, 1_000, 1_500, 2_000, 3_000];

export interface WaitOptions<TResult = RegimeRunResult> {
  /** Aborts the wait. The job itself is unaffected — it is running on the server. */
  signal?: AbortSignal;
  /** Called after each poll, so the page can show elapsed time or a request count. */
  onPoll?: (job: JobRecord<TResult>, attempt: number) => void;
}

export class JobAborted extends Error {
  constructor() {
    super("The wait for this job was cancelled.");
    this.name = "JobAborted";
  }
}

/**
 * Poll a job until it is no longer running, and return the terminal record.
 *
 * A `failed` job resolves rather than rejects. That is deliberate: the failure is a fact
 * about the run — a refusal the engine made, with a code and a hint — and the page needs
 * to render it in the same place it would render a result. Rejecting would push it into a
 * `catch` alongside transport errors, where the two would be indistinguishable.
 *
 * `TResult` names what the job's own kind carries — Module D's detection or C7's scaling —
 * so a page reads its result without narrowing a union the job registry already decided.
 */
export async function waitForJob<TResult = RegimeRunResult>(
  jobId: string,
  { signal, onPoll }: WaitOptions<TResult> = {},
): Promise<JobRecord<TResult>> {
  for (let attempt = 0; ; attempt += 1) {
    if (signal?.aborted === true) throw new JobAborted();

    const job = await api.job<TResult>(jobId);
    onPoll?.(job, attempt);
    if (job.status !== "running") return job;

    await sleep(BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)] ?? 3_000, signal);
  }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(new JobAborted());
    }
    if (signal?.aborted === true) {
      clearTimeout(timer);
      reject(new JobAborted());
      return;
    }
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** The message to show for a failed job. Falls back to the status line.
 *
 *  Typed over `unknown` because it reads only the envelope: both a detection failure and a
 *  scaling failure carry the same `error` shape, and a signature narrowed to one of them
 *  would be a signature the other could not call. */
export function failureMessage(job: JobRecord<unknown>): string {
  const error = job.error;
  if (error !== null) {
    const hint = error.details["hint"];
    return typeof hint === "string" && hint !== ""
      ? `${error.message} ${hint}`
      : error.message;
  }
  return job.message ?? "The run failed without saying why.";
}

/** True when an unknown thrown value is this module's cancellation, not a real failure. */
export function isAbort(cause: unknown): boolean {
  return cause instanceof JobAborted;
}
