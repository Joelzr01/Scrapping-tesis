# Herramienta para Revisión Sistemática de Literatura (RSL)

Esta es una aplicación desarrollada con Streamlit que permite realizar revisiones sistemáticas de literatura de manera interactiva. Utiliza Google Scholar para buscar artículos, permite filtrar por palabras clave y año, y genera visualizaciones como grafos de coautoría y agrupamientos por similitud temática.

## 🚀 Características

- Búsqueda de artículos en Google Scholar
- Filtrado por palabras clave y año
- Grafo de coautoría con NetworkX
- Agrupamiento de artículos por similitud de títulos (TF-IDF + KMeans)
- Exportación de artículos seleccionados como los "mejores" a CSV
- Visualización interactiva de relaciones entre artículos

## 📦 Requisitos

Asegúrate de que el archivo `requirements.txt` esté presente. Puedes instalar las dependencias ejecutando:

```bash
pip install -r requirements.txt
