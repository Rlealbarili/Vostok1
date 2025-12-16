-- ============================================================================
-- VOSTOK-1 DATABASE INITIALIZATION
-- Extensões necessárias: TimescaleDB + PGVector
-- ============================================================================

-- Habilitar extensões
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;

-- Confirmar instalação
SELECT extname, extversion FROM pg_extension 
WHERE extname IN ('timescaledb', 'vector');
