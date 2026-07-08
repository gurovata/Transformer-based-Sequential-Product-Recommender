CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS items (
    item_id INTEGER PRIMARY KEY,
    item_category TEXT NOT NULL,
    item_price_bucket TEXT NOT NULL,
    item_description TEXT,
    base_popularity REAL
);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    item_category TEXT,
    item_price_bucket TEXT,
    position INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (item_id) REFERENCES items(item_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    item_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (item_id) REFERENCES items(item_id)
);

