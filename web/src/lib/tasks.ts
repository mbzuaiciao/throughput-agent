import { readFileSync } from "node:fs";
import path from "node:path";

export type TaskSummary = {
  task_id: string;
  split: string;
  model: string;
  n_gpus: number;
  gpu_type: string;
  seq_len: number;
  global_batch_tokens: number;
  attempt_budget: number;
};

function parseSimpleTask(relativePath: string): TaskSummary {
  // Read only trusted, checked-in YAML task specifications.
  // Each task is currently a flat mapping with scalar values and comments.
  const raw = readFileSync(path.resolve(process.cwd(), relativePath), "utf-8");
  const map: Record<string, string> = {};
  for (const line of raw.split(/\r?\n/)) {
    const match = line.match(/^([a-z_]+):\s*([^#]*?)(?:\s*#.*)?$/);
    if (match) map[match[1]] = match[2].trim();
  }
  return {
    task_id: map.task_id,
    split: map.split,
    model: map.model,
    n_gpus: Number(map.n_gpus),
    gpu_type: map.gpu_type,
    seq_len: Number(map.seq_len),
    global_batch_tokens: Number(map.global_batch_tokens),
    attempt_budget: Number(map.attempt_budget),
  };
}

export const tasks = [
  parseSimpleTask("../tasks/dev/gpt2s_1gpu_a100.yaml"),
  parseSimpleTask("../tasks/heldout_shift/gpt2m_4gpu_a10.yaml"),
];
