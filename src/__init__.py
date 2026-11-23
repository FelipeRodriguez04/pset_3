# En Windows, si pyarrow (lectura de parquet) carga su runtime antes que LightGBM,
# LightGBM falla con "access violation" al construir el Dataset. Importarlo primero
# evita el conflicto de DLLs; en Linux/macOS no tiene efecto.
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass
