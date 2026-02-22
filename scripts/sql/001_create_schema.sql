CREATE TABLE IF NOT EXISTS flights (
  flight_id TEXT PRIMARY KEY,
  dep_airport TEXT NOT NULL,
  arr_airport TEXT NOT NULL,
  scheduled_dep TIMESTAMP NOT NULL,
  actual_dep TIMESTAMP,
  delay_minutes INT CHECK (delay_minutes >= 0),
  passenger_count INT CHECK (passenger_count >= 0),
  fuel_consumption_kg NUMERIC CHECK (fuel_consumption_kg >= 0),
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS humanitarian_shipments (
  shipment_id TEXT PRIMARY KEY,
  region TEXT NOT NULL,
  item_type TEXT NOT NULL,
  quantity INT NOT NULL CHECK (quantity >= 0),
  priority INT NOT NULL CHECK (priority BETWEEN 1 AND 5),
  status TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cern_events (
  event_id TEXT PRIMARY KEY,
  detector TEXT NOT NULL,
  energy_gev NUMERIC NOT NULL CHECK (energy_gev >= 0),
  is_rare_event BOOLEAN NOT NULL,
  recorded_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_flights_scheduled_dep ON flights (scheduled_dep);
CREATE INDEX IF NOT EXISTS idx_flights_dep_airport ON flights (dep_airport);
CREATE INDEX IF NOT EXISTS idx_flights_arr_airport ON flights (arr_airport);

CREATE INDEX IF NOT EXISTS idx_shipments_region ON humanitarian_shipments (region);
CREATE INDEX IF NOT EXISTS idx_shipments_priority ON humanitarian_shipments (priority);

CREATE INDEX IF NOT EXISTS idx_cern_events_recorded_at ON cern_events (recorded_at);
CREATE INDEX IF NOT EXISTS idx_cern_events_is_rare_event ON cern_events (is_rare_event);
