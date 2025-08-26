document.addEventListener('DOMContentLoaded', () => {
    // --- Vistas y elementos ---
    const mainView = document.getElementById('main-view');
    const testView = document.getElementById('test-view');
    const pdfUpload = document.getElementById('pdf-upload');
    const statusText = document.getElementById('status-text');
    const testTitle = document.getElementById('test-title');
    const testContainer = document.getElementById('test-container');
    const gradeTestBtn = document.getElementById('grade-test-btn');
    const newTestBtn = document.getElementById('new-test-btn');
    const backToMainBtn = document.getElementById('back-to-main-btn');
    const resultsContainer = document.getElementById('results-container');
    const feedbackContent = document.getElementById('feedback-content');
    const themeToggle = document.getElementById('checkbox');

    let documentChunks = []; // Almacenará los trozos de texto con sus vectores
    let currentTestQuestions = [];
    let embedder; // Almacenará el modelo de IA para que no se recargue

    // --- PASO 1: PROCESAR EL PDF ---
    pdfUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        testTitle.textContent = `Test sobre: ${file.name}`;
        statusText.textContent = 'Leyendo PDF...';
        
        try {
            const pdfText = await readPdf(file);
            statusText.textContent = 'Analizando el texto... (esto puede tardar un poco la primera vez)';
            
            // Troceamos el texto en frases o párrafos que tengan sentido
            const chunks = pdfText.match(/[^.!?]+[.!?]+/g) || [];
            
            // Inicializamos el modelo de IA la primera vez
            if (!embedder) {
                statusText.textContent = 'Cargando modelo de IA en el navegador...';
                embedder = await window.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
                    progress_callback: data => {
                        if (data.status === 'progress') {
                            statusText.textContent = `Descargando modelo... ${Math.round(data.progress)}%`;
                        }
                    }
                });
            }
            
            documentChunks = []; // Limpiamos los datos de PDFs anteriores
            for (let i = 0; i < chunks.length; i++) {
                const chunk = chunks[i].trim();
                // Ignoramos frases muy cortas que no dan para una pregunta
                if (chunk.split(' ').length < 10) continue; 
                
                const output = await embedder(chunk, { pooling: 'mean', normalize: true });
                documentChunks.push({ text: chunk, embedding: Array.from(output.data) });
                statusText.textContent = `Analizando texto... ${Math.round((i / chunks.length) * 100)}%`;
            }

            statusText.textContent = `¡PDF procesado! ${documentChunks.length} fragmentos listos.`;
            startTest();

        } catch (error) {
            console.error(error);
            statusText.textContent = 'Error al procesar el PDF.';
            alert('Hubo un error al procesar el PDF. Intenta con otro.');
        }
    });

    // Función auxiliar para leer el texto del PDF
    async function readPdf(file) {
        const pdfjsLib = window['pdfjs-dist/build/pdf'];
        pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.11.338/pdf.worker.min.js`;
        
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
        let fullText = '';
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            fullText += textContent.items.map(item => item.str).join(' ') + '\n';
        }
        return fullText;
    }

    // --- PASO 2: GENERAR Y MOSTRAR EL TEST ---
    function startTest() {
        mainView.classList.add('hidden');
        testView.classList.remove('hidden');
        
        currentTestQuestions = generateQuestions(10);
        if (currentTestQuestions.length < 10) {
            alert(`No se han podido generar 10 preguntas de calidad. Se han generado ${currentTestQuestions.length}. Prueba con un PDF más largo.`);
        }
        displayTest(currentTestQuestions);
    }

    function generateQuestions(numQuestions) {
        const questions = [];
        const usedChunks = new Set();

        while (questions.length < numQuestions && usedChunks.size < documentChunks.length) {
            const randomIndex = Math.floor(Math.random() * documentChunks.length);
            if (usedChunks.has(randomIndex)) continue;
            usedChunks.add(randomIndex);

            const sourceChunk = documentChunks[randomIndex];
            
            // Estrategia para encontrar una palabra clave (ej: un sustantivo largo)
            const words = sourceChunk.text.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"").split(/\s/);
            const keyword = words.find(w => w.length > 6 && /^[A-Z]/.test(w)); // Busca una palabra larga que empiece por mayúscula
            
            if (!keyword) continue; // Si no encuentra, prueba con otro chunk

            const questionText = sourceChunk.text.replace(keyword, '______');
            const correctAnswer = keyword;
            
            // Buscamos distractores (opciones incorrectas)
            const distractors = findDistractors(sourceChunk.embedding, correctAnswer);
            if (distractors.length < 3) continue; // Necesitamos 3 opciones falsas

            const options = [...new Set([correctAnswer, ...distractors])].sort(() => 0.5 - Math.random());
            if (options.length < 4) continue; // Aseguramos que tenemos 4 opciones únicas
            
            questions.push({
                pregunta: questionText,
                opciones: options,
                respuestaCorrecta: options.indexOf(correctAnswer)
            });
        }
        return questions;
    }

    function findDistractors(sourceEmbedding, correctAnswer) {
        const similarities = [];
        for (const chunk of documentChunks) {
            if (chunk.text.includes(correctAnswer)) continue;
            
            const words = chunk.text.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"").split(/\s/);
            const keyword = words.find(w => w.length > 6 && /^[A-Z]/.test(w));
            if (!keyword || keyword.toLowerCase() === correctAnswer.toLowerCase()) continue;

            const similarity = window.cos_sim(sourceEmbedding, chunk.embedding);
            similarities.push({ distractor: keyword, score: similarity });
        }

        return [...new Set(similarities.sort((a, b) => b.score - a.score)
                           .slice(0, 10) // Cogemos 10 para tener variedad
                           .map(item => item.distractor))]
                           .slice(0, 3); // Y nos quedamos con 3 únicos
    }

    function displayTest(questions) {
        testContainer.innerHTML = '';
        resultsContainer.classList.add('hidden');
        gradeTestBtn.style.display = 'block';

        questions.forEach((q, index) => {
            const questionBlock = document.createElement('div');
            questionBlock.className = 'question-block';
            const optionsHTML = q.opciones.map((opt, i) => `
                <li>
                    <input type="radio" name="question${index}" id="q${index}o${i}" value="${i}">
                    <label for="q${index}o${i}">${String.fromCharCode(65 + i)}) ${opt}</label>
                </li>
            `).join('');

            questionBlock.innerHTML = `<p>${index + 1}. ${q.pregunta}</p><ul class="options-list">${optionsHTML}</ul>`;
            testContainer.appendChild(questionBlock);
        });
    }

    function gradeTest() {
        gradeTestBtn.style.display = 'none';
        let score = 0;
        
        currentTestQuestions.forEach((q, index) => {
            const selectedOption = document.querySelector(`input[name="question${index}"]:checked`);
            
            document.getElementsByName(`question${index}`).forEach(radio => radio.disabled = true);
            
            const correctLabel = document.querySelector(`label[for="q${index}o${q.respuestaCorrecta}"]`);
            if (correctLabel) correctLabel.classList.add('correct');
            
            if (selectedOption) {
                if (parseInt(selectedOption.value) === q.respuestaCorrecta) {
                    score++;
                } else {
                    selectedOption.nextElementSibling.classList.add('incorrect');
                }
            }
        });

        resultsContainer.classList.remove('hidden');
        let feedbackHTML = `<h3>Has acertado ${score} de ${currentTestQuestions.length}.</h3>`;
        if (score < currentTestQuestions.length) {
            feedbackHTML += `<p>¡Sigue dándole caña a esos temas!</p>`;
        } else {
            feedbackHTML += `<p>¡Enhorabuena, estás hecho un máquina!</p>`;
        }
        feedbackContent.innerHTML = feedbackHTML;
    }

    // --- EVENT LISTENERS Y LÓGICA DE UI ---
    newTestBtn.addEventListener('click', startTest);
    gradeTestBtn.addEventListener('click', gradeTest);
    backToMainBtn.addEventListener('click', () => {
        documentChunks = [];
        pdfUpload.value = '';
        statusText.textContent = '';
        mainView.classList.remove('hidden');
        testView.classList.add('hidden');
    });

    // Modo oscuro
    themeToggle.addEventListener('change', (e) => {
        document.body.classList.toggle('dark-mode', e.target.checked);
        localStorage.setItem('theme', e.target.checked ? 'dark-mode' : 'light');
    });
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme === 'dark-mode') {
        document.body.classList.add('dark-mode');
        themeToggle.checked = true;
    }

    // Carga inicial
    console.log("Consti-Tutor Test PRO listo.");
});
