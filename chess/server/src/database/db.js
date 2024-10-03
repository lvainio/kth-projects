import sqlite3 from "sqlite3";
import resolvePath from "../util.js";

sqlite3.verbose();

const path = resolvePath("server", "src", "database", "sqlite.db");
const db = new sqlite3.Database(path);

db.serialize(() => {
  // DROP TABLES
  db.run(`DROP TABLE IF EXISTS user`);
  db.run(`DROP TABLE IF EXISTS game`);

  // CREATE TABLES
  db.run(`PRAGMA foreign_keys = ON`);

  db.run(`CREATE TABLE IF NOT EXISTS user (
    user_id     INTEGER PRIMARY KEY,
    username    TEXT NOT NULL,
    password    TEXT NOT NULL)`);

  db.run(`CREATE TABLE IF NOT EXISTS game (
    game_id         INTEGER     PRIMARY KEY,
    white           INTEGER     NOT NULL REFERENCES user (user_id),
    black           INTEGER     NOT NULL REFERENCES user (user_id),
    result          TEXT        CHECK (result IN ('BLACK','WHITE','DRAW')) NOT NULL DEFAULT 'DRAW')`);
});

export default db;
