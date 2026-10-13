// Reads raw_specs.json and turns each spec into a real training example using
// the actual buildPrompt() from the Esnad repo — so training data never drifts
// from the real production prompt format.
const fs = require("fs");
const path = require("path");
const { buildPrompt } = require("C:/Users/kimat/GP2-Cl/services/aiEngine");

const specs = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "datasets", "raw_specs.json"), "utf-8"));

const lines = specs.map((spec) => {
  const prompt = buildPrompt({
    userPrompt: spec.userPrompt,
    projectIdea: spec.projectIdea,
    teamMembers: spec.teamMembers,
  });
  const completion = JSON.stringify(spec.expectedTasks);
  return JSON.stringify({ prompt, completion });
});

fs.writeFileSync(path.join(__dirname, "..", "datasets", "sft_data.jsonl"), lines.join("\n") + "\n", "utf-8");
console.log(`Wrote ${lines.length} training examples to datasets/sft_data.jsonl`);
