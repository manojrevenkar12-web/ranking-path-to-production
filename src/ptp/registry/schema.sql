-- The model registry and the record of the path to production. Stage transitions, gate results, deployments and
-- relay messages are append-only: corrections are new rows.

CREATE TABLE IF NOT EXISTS model_seq (
    kind text PRIMARY KEY,
    next integer NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id text PRIMARY KEY,
    kind text NOT NULL,
    engine text NOT NULL,
    started_at bigint NOT NULL,
    finished_at bigint,
    state text NOT NULL,              -- RUNNING, SUCCEEDED, FAILED
    outcome text,                     -- candidate, blocked, no_data, error
    version text,
    params jsonb NOT NULL DEFAULT '{}',
    error text
);

CREATE TABLE IF NOT EXISTS model_versions (
    version text PRIMARY KEY,
    kind text NOT NULL,               -- ranker, tower
    created_at bigint NOT NULL,
    pipeline_run text NOT NULL REFERENCES pipeline_runs(run_id),
    code_commit text NOT NULL,
    data_snapshot text NOT NULL,
    training jsonb NOT NULL,
    features jsonb NOT NULL,
    artifacts jsonb NOT NULL,         -- name -> {key, sha256}
    metrics jsonb NOT NULL,
    dataset_profile jsonb NOT NULL,
    model_card jsonb NOT NULL,
    owner text NOT NULL,
    status text NOT NULL              -- rejected, candidate, staging, canary, production, archived, rolled_back, failed
);

CREATE TABLE IF NOT EXISTS stage_transitions (
    id bigserial PRIMARY KEY,
    version text NOT NULL REFERENCES model_versions(version),
    from_status text,
    to_status text NOT NULL,
    at bigint NOT NULL,
    actor text NOT NULL,
    reason text NOT NULL
);

CREATE TABLE IF NOT EXISTS gate_results (
    id bigserial PRIMARY KEY,
    version text,                     -- null when the run stopped before a model existed
    pipeline_run text,
    stage text NOT NULL,              -- data, model, export, serving, canary
    gate text NOT NULL,
    passed boolean NOT NULL,
    severity text NOT NULL,
    blocking boolean NOT NULL,
    waived_by text,
    summary text NOT NULL,
    evidence jsonb NOT NULL,
    at bigint NOT NULL
);

CREATE TABLE IF NOT EXISTS deployments (
    id bigserial PRIMARY KEY,
    version text NOT NULL REFERENCES model_versions(version),
    trigger text NOT NULL,            -- the dispatch or schedule that started it
    started_at bigint NOT NULL,
    finished_at bigint,
    state text NOT NULL,              -- running, canary, promoted, rolled_back, failed, waiting_embeddings
    steps jsonb NOT NULL DEFAULT '[]',
    package_sha text,
    error text
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_deployment_per_version
    ON deployments(version) WHERE state IN ('running', 'canary', 'waiting_embeddings');

CREATE TABLE IF NOT EXISTS packages (
    env text NOT NULL,                -- staging, prod
    generation integer NOT NULL,
    deployed_at bigint NOT NULL,
    sha256 text NOT NULL,
    profiles jsonb NOT NULL,
    zip bytea NOT NULL,
    PRIMARY KEY (env, generation)
);

CREATE TABLE IF NOT EXISTS routing (
    surface text PRIMARY KEY,         -- search, home
    production text NOT NULL,         -- profile (search) or tower version (home)
    canary text,
    share double precision NOT NULL DEFAULT 0,
    canary_since bigint,
    previous text,
    updated_at bigint NOT NULL
);

CREATE TABLE IF NOT EXISTS routing_log (
    id bigserial PRIMARY KEY,
    surface text NOT NULL,
    production text NOT NULL,
    canary text,
    share double precision NOT NULL,
    at bigint NOT NULL,
    reason text NOT NULL
);

CREATE TABLE IF NOT EXISTS relay_messages (
    message_id text PRIMARY KEY,
    received_at bigint NOT NULL,
    pipeline_run text,
    event jsonb NOT NULL,
    action text NOT NULL,             -- dispatched, ignored, duplicate_run
    dispatched_at bigint
);

CREATE TABLE IF NOT EXISTS monitor_runs (
    id bigserial PRIMARY KEY,
    surface text NOT NULL,
    version text,
    at bigint NOT NULL,
    healthy boolean NOT NULL,
    result jsonb NOT NULL
);

-- Lake rows an operator has declared bad (a feed or logging incident); training sets leave them out.
CREATE TABLE IF NOT EXISTS quarantine (
    id bigserial PRIMARY KEY,
    table_name text NOT NULL,
    day date NOT NULL,
    market text,                      -- null: every market
    reason text NOT NULL,
    by_whom text NOT NULL,
    at bigint NOT NULL
);

CREATE OR REPLACE FUNCTION refuse_change() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'append-only table %', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['stage_transitions', 'gate_results', 'routing_log', 'relay_messages'] LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = t || '_append_only') THEN
            EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION refuse_change()', t || '_append_only', t);
        END IF;
    END LOOP;
END $$;
