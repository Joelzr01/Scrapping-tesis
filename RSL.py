import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scholarly import scholarly
import streamlit as st
import pandas as pd
import io
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import mplcursors

def buscar_articulos(consulta, limite=10):
    resultados = []
    busqueda = scholarly.search_pubs(consulta)
    for _ in range(limite):
        try:
            resultados.append(next(busqueda))
        except StopIteration:
            break
    return resultados

def crear_grafo(articulos, etiquetas=None):
    G = nx.Graph()
    revista_por_autor = {}
    autor_por_articulo = []

    for i, art in enumerate(articulos):
        titulo = art.get("bib", {}).get("title", "Sin título")
        autores = art.get("bib", {}).get("author", [])
        revista = art.get("bib", {}).get("venue", "No especificada")

        if isinstance(autores, str):
            autores = [a.strip() for a in autores.split(" and ")]

        autor_por_articulo.append(autores)

        for i, autor1 in enumerate(autores):
            G.add_node(autor1)
            revista_por_autor.setdefault(autor1, []).append(revista)

            for autor2 in autores[i+1:]:
                if G.has_edge(autor1, autor2):
                    G[autor1][autor2]['coautoria'] = True
                else:
                    G.add_edge(autor1, autor2, coautoria=True)


    # Relaciones por cluster de contenido (si hay etiquetas)
    if etiquetas is not None:
        clusters = {}
        for idx, autores in enumerate(autor_por_articulo):
            grupo = etiquetas[idx]
            clusters.setdefault(grupo, set()).update(autores)

        for autores_grupo in clusters.values():
            autores_grupo = list(autores_grupo)
            for i in range(len(autores_grupo)):
                for j in range(i+1, len(autores_grupo)):
                    a1, a2 = autores_grupo[i], autores_grupo[j]
                    if G.has_edge(a1, a2):
                        G[a1][a2]['similitud_contenido'] = True
                    else:
                        G.add_edge(a1, a2, similitud_contenido=True)

    return G



def mostrar_grafo(G):
    st.subheader("Grafo de relaciones por coautoría y contenido")

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 9))

    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=600)
    nx.draw_networkx_labels(G, pos, font_size=9)

    # Coautoría
    edges_coautoria = [(u, v) for u, v, d in G.edges(data=True) if d.get("coautoria")]
    nx.draw_networkx_edges(G, pos, edgelist=edges_coautoria, edge_color='black', width=2)

    # Similitud de contenido
    edges_similitud = [(u, v) for u, v, d in G.edges(data=True) if d.get("similitud_contenido")]
    nx.draw_networkx_edges(G, pos, edgelist=edges_similitud, edge_color='blue', style='dotted', width=1.5)

    # Leyenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Coautoría'),
        Line2D([0], [0], color='blue', lw=2, linestyle='dotted', label='Similitud de contenido'),
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.title("Grafo de relaciones entre autores")
    st.pyplot(plt)



def agrupar_por_embeddings(titulos):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(titulos)
    kmeans = KMeans(n_clusters=min(4, len(titulos)), random_state=42).fit(X)
    pca = PCA(n_components=2)
    puntos = pca.fit_transform(X.toarray())
    return puntos, kmeans.labels_

def mostrar_cluster(puntos, etiquetas, titulos):
    st.subheader("Agrupamiento por similitud de título")
    fig, ax = plt.subplots(figsize=(12, 8))

    num_clusters = len(set(etiquetas))
    colores = cm.get_cmap("tab10", num_clusters)
    marcadores = ['o', 's', 'D', '^', 'P', '*', 'X', 'v', '<', '>']  # hasta 10 clusters diferentes

    puntos_anotados = []

    for i in range(num_clusters):
        idx = np.where(etiquetas == i)
        cluster_puntos = puntos[idx]
        cluster_titulos = [titulos[j] for j in idx[0]]

        scatter = ax.scatter(cluster_puntos[:, 0], cluster_puntos[:, 1],
                             label=f"Cluster {i + 1}",
                             c=[colores(i)], marker=marcadores[i % len(marcadores)], s=100, alpha=0.6)

        for j, (x, y) in enumerate(cluster_puntos):
            titulo = cluster_titulos[j]
            corto = (titulo[:50] + '...') if len(titulo) > 50 else titulo
            ax.text(x, y, corto, fontsize=7)
            puntos_anotados.append((x, y, titulo))

    ax.set_title("Agrupamiento por embeddings (TF-IDF + KMeans)")
    ax.legend(title="Grupos", loc="best")
    ax.grid(True)

    cursor = mplcursors.cursor(ax, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        for px, py, titulo in puntos_anotados:
            if np.isclose([px], [x], atol=0.05) and np.isclose([py], [y], atol=0.05):
                sel.annotation.set(text=titulo)
                break

    st.pyplot(fig)


def app():
    st.title("Herramienta para Revisión Sistemática de Literatura (RSL)")

    if "articulos_filtrados" not in st.session_state:
        st.session_state.articulos_filtrados = []
    if "busqueda_realizada" not in st.session_state:
        st.session_state.busqueda_realizada = False

    consulta = st.text_input("Introduce la temática de búsqueda")
    limite = st.slider("Número de artículos a obtener", 5, 30, 10)
    palabras_clave = st.text_input("Filtrar artículos por palabras clave (separadas por coma)")
    años = st.slider("Filtrar artículos desde los últimos N años", 1, 10, 5)
    año_min = datetime.now().year - años

    if st.button("Buscar artículos"):
        articulos = buscar_articulos(consulta, limite)

        # Filtrar artículos que tengan pub_year válido y estén en el rango
        articulos = [
            art for art in articulos
            if str(art["bib"].get("pub_year", "")).isdigit() and int(art["bib"]["pub_year"]) >= año_min
        ]

        # Filtro por palabras clave si se proporcionan
        if palabras_clave:
            claves = [k.strip().lower() for k in palabras_clave.split(",")]
            articulos = [
                art for art in articulos
                if any(clave in art["bib"].get("title", "").lower() for clave in claves)
            ]

        st.session_state.articulos_filtrados = articulos
        st.session_state.busqueda_realizada = True

    if st.session_state.busqueda_realizada and st.session_state.articulos_filtrados:
        articulos_filtrados = st.session_state.articulos_filtrados

        st.subheader("Artículos encontrados")
        for idx, art in enumerate(articulos_filtrados):
            bib = art.get("bib", {})
            resumen = bib.get("abstract", None)
            if not resumen:
                resumen = "Resumen no disponible. [Resumen generado automáticamente: Artículo sobre temática relacionada con la búsqueda.]"

            titulo = bib.get('title', 'Sin título')
            enlace_google = f"https://scholar.google.com/scholar?q={titulo.replace(' ', '+')}"
            st.markdown(f"**{idx+1}. [{titulo}]({enlace_google})**")
            st.markdown(f"- **Autores**: {bib.get('author', 'Desconocido')}")
            st.markdown(f"- **Año**: {bib.get('pub_year', 'N/A')}")
            st.markdown(f"- **Revista**: {bib.get('venue', 'No especificada')}")
            st.markdown(f"- **Resumen**: {resumen}")
            st.markdown("---")

        #G = crear_grafo(articulos_filtrados)
       #mostrar_grafo(G)

        # Agrupamiento con embeddings
        titulos = [art["bib"]["title"] for art in articulos_filtrados]
        puntos, etiquetas = agrupar_por_embeddings(titulos)
        G = crear_grafo(articulos_filtrados, etiquetas)
        mostrar_grafo(G)

        mostrar_cluster(puntos, etiquetas, titulos)
        

        st.subheader("Selecciona los mejores artículos")
        seleccionados = st.multiselect("Selecciona títulos relevantes:", titulos)

        if seleccionados:
            mejores = [
                art for art in articulos_filtrados
                if art["bib"]["title"] in seleccionados
            ]

            st.success("Has seleccionado los siguientes artículos como los mejores:")
            for art in mejores:
                st.write(f"- {art['bib'].get('title')}")

            # Crear DataFrame
            df_mejores = pd.DataFrame([{
                "Título": art["bib"].get("title", ""),
                "Autores": art["bib"].get("author", ""),
                "Año": art["bib"].get("pub_year", ""),
                "Revista": art["bib"].get("venue", "No especificada"),
                "Resumen": art["bib"].get("abstract", "Resumen no disponible"),
                
            } for art in mejores])

            # CSV
            csv = df_mejores.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar en CSV",
                data=csv,
                file_name='mejores_articulos.csv',
                mime='text/csv'
            )

    elif st.session_state.busqueda_realizada:
        st.warning("No se encontraron artículos con esas palabras clave o dentro del rango de años.")

if __name__ == "__main__":
    app()


