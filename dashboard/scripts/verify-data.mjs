import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";

const dataDir = new URL("../src/data/", import.meta.url);
const files = (await readdir(dataDir)).filter((file) => file.endsWith("_query_metrics.json"));
const sources = await Promise.all(files.map(async (file) => ({ file, records: JSON.parse(await readFile(new URL(file, dataDir), "utf8")) })));
const records = sources.flatMap(({ file, records: source }) => source.map((record) => ({ ...record, source: file })));

const expectedQueries = {
  animals: ["Q1_zebra_image_count", "Q2_lion_audio_count", "Q3_top_zebra_city", "Q10_top_zebra_city_station"],
  artwork: ["A1_madonna_and_child", "A2_more_than_two_people", "A3_more_than_three_people", "A4_saints_with_halos", "A5_death_dominant_theme", "A6_madonna_and_child_century"],
  medical: ["Q1_allergy_patients", "Q3_cancer_family_lung_problems_patients", "Q4_average_age_acne_patients", "Q9_sick_patients_skin_xray"],
  medical_online: ["Q1_allergy_patients", "Q3_cancer_family_lung_problems_patients", "Q4_average_age_acne_patients", "Q9_sick_patients_skin_xray"],
  movie: ["Q1_positive_reviews", "Q2_positive_reviews_for_taken_3", "Q3_reviews_same_sentiment", "Q4_positive_reviews_for_taken_3_ratio"],
  rotowire: ["R1_player_points_assists", "R2_player_points_home_team_won"],
};

const assertCoverage = (condition, message) => assert(condition, message);
assert.equal(records.length, 592, "Expected local plus online benchmark records");
assert.equal(files.length, 8, "Expected five local sources plus online Medical, Movie, and RotoWire");

const keyFor = (record) => `${record.source}:${record.query}:${record.model}:${record.run}`;
const keys = records.map(keyFor);
assert.equal(new Set(keys).size, keys.length, "Duplicate dataset/query/model/run records found");

for (const record of records) {
  assertCoverage(typeof record.runtime_seconds === "number" && record.runtime_seconds >= 0, `Missing runtime: ${keyFor(record)}`);
  assertCoverage(typeof record.energy_kwh === "number" && record.energy_kwh >= 0, `Missing energy: ${keyFor(record)}`);
  assertCoverage(typeof record.co2_g === "number" && record.co2_g >= 0, `Missing CO2: ${keyFor(record)}`);
  assertCoverage(typeof record.total_tokens === "number" && record.total_tokens >= 0, `Missing tokens: ${keyFor(record)}`);
  assertCoverage(Array.isArray(record.physical_operators) && record.physical_operators.length > 0, `Missing physical operators: ${keyFor(record)}`);
  assertCoverage(typeof record.quality?.metric === "string", `Missing quality contract: ${keyFor(record)}`);
}

for (const source of sources) {
  const demo = source.records[0]?.demo;
  assertCoverage(expectedQueries[demo], `Unknown source demo: ${demo}`);
  const observedQueries = [...new Set(source.records.map((record) => record.query))].sort();
  assert.deepEqual(observedQueries, [...expectedQueries[demo]].sort(), `Query coverage mismatch in ${source.file}`);
  const combinations = new Map();
  for (const record of source.records) {
    const key = `${record.query}:${record.model}`;
    const runs = combinations.get(key) ?? new Set();
    runs.add(String(record.run));
    combinations.set(key, runs);
  }
  for (const [key, runs] of combinations) {
    const expectedRuns = source.file.includes("rotowire_online") && key.endsWith(":gpt-4.1") ? 6 : 10;
    assert.equal(runs.size, expectedRuns, `Expected ${expectedRuns} runs for ${source.file} ${key}`);
  }
}

const count = (predicate) => records.filter(predicate).length;
const coverage = (label, value) => `${label}: ${value}/${records.length} (${Math.round((value / records.length) * 100)}%)`;
const phaseRecords = count((record) => (record.planner_energy_kwh ?? record.planning?.energy_kwh) != null && (record.execution_energy_kwh ?? record.execution?.energy_kwh) != null);
const qualityContracts = new Set(records.map((record) => record.quality.metric));
const throughputRecords = count((record) => typeof record.throughput === "number" || typeof record.throughput_rows_per_second === "number");
const structuredPlanRecords = count((record) => record.demo === "movie" && record.execution_plan?.logical?.steps?.length > 0 && record.execution_plan?.physical?.steps?.length > 0 && record.execution_plan?.edges?.length > 0);

assert.equal(qualityContracts.size, 10, "Unexpected quality contract count");
assert.equal(structuredPlanRecords, 160, "Expected structured logical/physical plans for all Movie records");
console.log(`Verified ${records.length} benchmark records from ${files.length} JSON sources.`);
console.log(`Core totals: ${coverage("runtime", count((record) => typeof record.runtime_seconds === "number"))}, ${coverage("energy", count((record) => typeof record.energy_kwh === "number"))}, ${coverage("CO2", count((record) => typeof record.co2_g === "number"))}, ${coverage("tokens", count((record) => typeof record.total_tokens === "number"))}.`);
console.log(`Phase fields: ${coverage("planner + execution", phaseRecords)}. Structured Movie plans: ${structuredPlanRecords}/160. Quality contracts: ${qualityContracts.size} dataset-specific contracts.`);
console.log(`Throughput: ${throughputRecords ? coverage("recorded", throughputRecords) : `unsupported (0/${records.length}; no source field)`}.`);
