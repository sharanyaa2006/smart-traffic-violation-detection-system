-- Smart Traffic Violation Detection System
-- Run this once to set up the database before running the app.

CREATE DATABASE IF NOT EXISTS traffic_violations;
USE traffic_violations;

CREATE TABLE IF NOT EXISTS violations (
    violation_id INT AUTO_INCREMENT PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    timestamp DATETIME NOT NULL,
    track_id INT,
    confidence FLOAT,
    snapshot_path VARCHAR(255),
    plate_number VARCHAR(20),
    video_source VARCHAR(255)
);

-- Useful for the analytics tab later (filter/sort by type or date fast)
CREATE INDEX idx_violation_type ON violations(type);
CREATE INDEX idx_violation_timestamp ON violations(timestamp);