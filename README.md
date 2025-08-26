# 📝 Consti-Tutor Test PRO

Una herramienta web para convertir cualquier PDF en un test de autoevaluación al instante. Ideal para chapar apuntes, leyes o lo que te echen.

## ¿Cómo funciona esta movida?

Esta aplicación es pura magia del frontend. Usa librerías que se ejecutan directamente en tu navegador para hacer todo el trabajo sin necesidad de un servidor detrás.

-   **Lectura de PDF**: Gracias a **PDF.js**, la aplicación lee y extrae todo el texto de tu documento.
-   **Inteligencia Artificial en el Navegador**: Con **Transformers.js** y el modelo `Xenova/all-MiniLM-L6-v2`, la herramienta analiza los fragmentos de texto, los convierte en vectores y encuentra las palabras clave y los distractores más adecuados para generar las preguntas del test.

Todo el proceso se realiza en tu máquina, garantizando que tus documentos son solo tuyos.

## 🚀 Cómo usarlo

1.  Abre la página web.
2.  Pulsa en **"Seleccionar PDF"** y elige el documento que quieras estudiar.
3.  Espera a que la IA procese el texto (la primera vez puede tardar un poco mientras descarga el modelo).
4.  ¡Listo! Responde a las preguntas del test que se ha generado.
5.  Cuando termines, pulsa en **"Corregir Test"** para ver tu nota.

## 🛠️ Despliegue

Este proyecto es una aplicación web estática (HTML, CSS, JS). Se puede desplegar fácilmente en cualquier servicio de alojamiento estático como:

-   GitHub Pages
-   Netlify
-   Vercel

No requiere configuración de backend.

---
Creado con cafeína y ganas de aprobar. ¡A darle caña!
