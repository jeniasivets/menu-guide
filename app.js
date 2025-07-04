// Global variables
let tokenizer = null;
let textModel = null;
let selectedImage = null;

// DOM elements
const uploadArea = document.getElementById('upload-area');
const menuImageInput = document.getElementById('menu-image-input');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const removeImageBtn = document.getElementById('remove-image');
const openaiKeyInput = document.getElementById('openai-key');
const clearKeyBtn = document.getElementById('clear-key-btn');
const processBtn = document.getElementById('process-image-btn');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const resultsSection = document.getElementById('results-section');
const resultsContainer = document.getElementById('results-container');
const consoleOutput = document.getElementById('console-output');

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    log('🚀 AI Menu Visualizer initialized', 'info');
    setupEventListeners();
    loadApiKey();
});

// Setup event listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => menuImageInput.click());
    
    // File input change
    menuImageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Remove image
    removeImageBtn.addEventListener('click', removeImage);
    
    // API key clear button
    clearKeyBtn.addEventListener('click', clearApiKey);
    
    // Process button
    processBtn.addEventListener('click', processImage);
    
    // API key input
    openaiKeyInput.addEventListener('input', updateProcessButton);
    
    // Keyboard shortcuts
    uploadArea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') menuImageInput.click();
    });
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        if (file.size > 20 * 1024 * 1024) {
            alert('File size must be less than 20MB');
            return;
        }
        const reader = new FileReader();
        reader.onload = function(e) {
            selectedImage = e.target.result;
            previewImg.src = selectedImage;
            imagePreview.style.display = 'block';
            uploadArea.style.display = 'none';
            updateProcessButton();
            log('📸 Image uploaded successfully', 'success');
        };
        reader.readAsDataURL(file);
    }
}

// Handle drag and drop
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    if (event.dataTransfer.files.length > 0) handleFileSelect({ target: { files: event.dataTransfer.files } });
}

// Remove image
function removeImage() {
    selectedImage = null;
    imagePreview.style.display = 'none';
    uploadArea.style.display = 'block';
    menuImageInput.value = '';
    updateProcessButton();
    log('🗑️ Image removed', 'info');
}

// Clear API key
function clearApiKey() {
    openaiKeyInput.value = '';
    sessionStorage.removeItem('openai_api_key');
    updateProcessButton();
    log('🔑 API key cleared', 'info');
}

// Load API key from session storage
function loadApiKey() {
    const savedKey = sessionStorage.getItem('openai_api_key');
    if (savedKey) {
        openaiKeyInput.value = savedKey;
        updateProcessButton();
    }
}

// Update process button state
function updateProcessButton() {
    const hasImage = selectedImage !== null;
    const hasApiKey = openaiKeyInput.value.trim() !== '';
    processBtn.disabled = !(hasImage && hasApiKey);
}

// Process image
async function processImage() {
    if (!selectedImage || !openaiKeyInput.value.trim()) {
        log('❌ Please upload an image and enter your API key', 'error');
        return;
    }
    
    // Save API key to session storage
    sessionStorage.setItem('openai_api_key', openaiKeyInput.value.trim());
    
    // Show loading
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    processBtn.disabled = true;
    
    try {
        log('🔍 Starting image analysis...', 'info');
        
        // Step 1: Extract text using OpenAI Vision API
        loadingText.textContent = 'Extracting text from image...';
        const extractedText = await extractTextFromImage(selectedImage);
        
        if (!extractedText || extractedText.trim() === '') {
            throw new Error('No text could be extracted from the image');
        }
        
        log('✅ Text extracted successfully', 'success');
        log(`📝 Extracted text: ${extractedText}`, 'info');
        
        // Step 2: Parse dish names
        loadingText.textContent = 'Parsing dish names...';
        const dishNames = await parseDishNames(extractedText);
        
        if (!dishNames || dishNames.length === 0) {
            throw new Error('No dish names could be identified');
        }
        
        log(`🍽️ Found ${dishNames.length} dish names`, 'success');
        dishNames.forEach(dish => {
            log(`   - ${dish.original} → ${dish.translated}`, 'info');
        });
        
        // Step 3: Generate embeddings and find similar dishes
        loadingText.textContent = 'Loading CLIP model...';
        await loadCLIPModel();
        
        loadingText.textContent = 'Finding similar dishes...';
        const results = await findSimilarDishes(dishNames);
        
        // Step 4: Display results
        displayResults(results);
        
        log('🎉 Processing completed successfully!', 'success');
        
    } catch (error) {
        log(`❌ Error: ${error.message}`, 'error');
        console.error('Processing error:', error);
    } finally {
        loading.style.display = 'none';
        processBtn.disabled = false;
        updateProcessButton();
    }
}

// Extract text from image using OpenAI Vision API
async function extractTextFromImage(imageData) {
    const apiKey = openaiKeyInput.value.trim();
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: 'gpt-4.1-nano',
            messages: [
                {
                    role: 'user',
                    content: [
                        {
                            type: 'text',
                            text: 'Extract all dish names from this menu image. Return only the dish names, one per line, in the original language. Do not include prices, descriptions, or any other text. Each dish name should be on its own line.'
                        },
                        {
                            type: 'image_url',
                            image_url: {
                                url: imageData
                            }
                        }
                    ]
                }
            ],
            max_tokens: 500
        })
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(`OpenAI API error: ${error.error?.message || 'Unknown error'}`);
    }
    const data = await response.json();
    return data.choices[0].message.content.trim();
}

// Parse dish names and translate to English (robust JSON extraction)
async function parseDishNames(extractedText) {
    const apiKey = openaiKeyInput.value.trim();
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: 'gpt-4.1-nano',
            messages: [
                {
                    role: 'user',
                    content: `Parse the following dish names from a menu and translate them to English. Return a JSON array with objects containing "original" (original text) and "translated" (English translation) fields. Only include actual dish names, not prices or descriptions.\n\nDish names:\n${extractedText}\n\nReturn only valid JSON, no other text. Example format:\n[{"original": "Salade Lyonnaise", "translated": "Lyonnaise Salad"}, {"original": "Oeuf mollette", "translated": "Soft-Boiled Egg"}]`
                }
            ],
            max_tokens: 1000
        })
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(`OpenAI API error: ${error.error?.message || 'Unknown error'}`);
    }
    const data = await response.json();
    const content = data.choices[0].message.content.trim();
    // Try to extract JSON array from the response
    const match = content.match(/\[.*\]/s);
    if (match) {
        try {
            return JSON.parse(match[0]);
        } catch (e) {
            log('⚠️ JSON extraction failed, falling back to line split...', 'warning');
        }
    }
    // Fallback: treat only lines that look like dish names as dish names
    const lines = content.split('\n').map(line => line.trim()).filter(line =>
        line && !['[', ']', '{', '}', ','].includes(line) && !line.startsWith('"original"') && !line.startsWith('"translated"')
    );
    return lines.map(line => ({
        original: line,
        translated: line
    }));
}

// Load CLIP model
async function loadCLIPModel() {
    if (tokenizer && textModel) return;
    
    log('📦 Loading CLIP models...', 'info');
    
    try {
        const { AutoTokenizer, CLIPTextModelWithProjection } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
        
        log('Step 1: Loading AutoTokenizer...', 'info');
        tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch32', { quantized: true });
        log('✅ AutoTokenizer loaded successfully', 'success');
        
        log('Step 2: Loading CLIPTextModelWithProjection...', 'info');
        textModel = await CLIPTextModelWithProjection.from_pretrained('Xenova/clip-vit-base-patch32', { quantized: true });
        log('✅ CLIPTextModelWithProjection loaded successfully', 'success');
        
        log('🎉 All CLIP models loaded successfully!', 'success');
    } catch (error) {
        log(`❌ Error loading CLIP models: ${error.message}`, 'error');
        throw error;
    }
}

// Fetch key ingredients for a dish using OpenAI API
async function fetchKeyIngredients(dishName, apiKey) {
    const prompt = `List the key ingredients for the dish "${dishName}" as a short, comma-separated list. Only list the main ingredients, no instructions or extra text.`;
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: 'gpt-4.1-nano',
            messages: [
                { role: 'user', content: prompt }
            ],
            max_tokens: 100
        })
    });
    if (!response.ok) {
        const error = await response.json();
        log(`❌ Ingredient API error for ${dishName}: ${error.error?.message || 'Unknown error'}`, 'error');
        return 'Could not fetch ingredients';
    }
    const data = await response.json();
    // Just return the content as-is (should be a comma-separated list)
    return data.choices[0].message.content.trim();
}

// Find similar dishes and fetch ingredients for each dish
async function findSimilarDishes(dishNames) {
    const apiKey = openaiKeyInput.value.trim();
    const results = [];
    for (const dish of dishNames) {
        log(`🔍 Finding similar dishes for: ${dish.translated}`, 'info');
        // Generate text embedding
        const textEmbedding = await generateTextEmbedding(dish.translated);
        // Find similar dishes
        const similarDishes = findTopSimilarDishes(textEmbedding, 3);
        // Fetch key ingredients from API
        let keyIngredients = '';
        try {
            keyIngredients = await fetchKeyIngredients(dish.translated, apiKey);
        } catch (e) {
            keyIngredients = 'Could not fetch ingredients';
        }
        results.push({
            original: dish.original,
            translated: dish.translated,
            similarDishes: similarDishes,
            keyIngredients: keyIngredients
        });
    }
    return results;
}

// Find top similar dishes (array logic, use match.image_path)
function findTopSimilarDishes(textEmbedding, topK = 3) {
    log(`🔍 Comparing text embedding (${textEmbedding.length} dimensions) with ${dishEmbeddings.length} dish embeddings`, 'info');
    const similarities = dishEmbeddings.map(dish => ({
        ...dish,
        similarity: cosineSimilarity(textEmbedding, dish.embedding)
    }));
    similarities.sort((a, b) => b.similarity - a.similarity);
    return similarities.slice(0, topK);
}

// Calculate cosine similarity
function cosineSimilarity(vecA, vecB) {
    // Ensure both vectors are arrays
    const arrayA = Array.isArray(vecA) ? vecA : Array.from(vecA);
    const arrayB = Array.isArray(vecB) ? vecB : Array.from(vecB);
    
    // Check if vectors have the same length
    if (arrayA.length !== arrayB.length) {
        console.error('Vector length mismatch:', arrayA.length, 'vs', arrayB.length);
        return 0;
    }
    
    const dotProduct = arrayA.reduce((sum, a, i) => sum + a * arrayB[i], 0);
    const normA = Math.sqrt(arrayA.reduce((sum, a) => sum + a * a, 0));
    const normB = Math.sqrt(arrayB.reduce((sum, b) => sum + b * b, 0));
    
    // Avoid division by zero
    if (normA === 0 || normB === 0) {
        return 0;
    }
    
    return dotProduct / (normA * normB);
}

// Generate text embedding
async function generateTextEmbedding(text) {
    if (!tokenizer || !textModel) {
        throw new Error('CLIP models not loaded');
    }
    
    // Tokenize the text
    const textInputs = tokenizer([text], { padding: true, truncation: true });
    
    // Generate embeddings
    const { text_embeds } = await textModel(textInputs);
    
    // Convert to array and return
    return text_embeds.tolist()[0];
}

// Display results (use match.image_path for images, clean output)
function displayResults(results) {
    resultsContainer.innerHTML = '';
    results.forEach(result => {
        const resultGroup = document.createElement('div');
        resultGroup.className = 'result-group';
        const title = document.createElement('h3');
        title.innerHTML = `<strong>${result.translated}</strong> <em>(${result.original})</em>`;
        const ingredientsDiv = document.createElement('div');
        ingredientsDiv.className = 'ingredients-list';
        ingredientsDiv.innerHTML = `<strong>Key Ingredients:</strong> ${result.keyIngredients}`;
        const imagesDiv = document.createElement('div');
        imagesDiv.className = 'result-images';
        result.similarDishes.forEach(match => {
            const imageWrapper = document.createElement('div');
            imageWrapper.className = 'result-image-wrapper';
            const img = document.createElement('img');
//            img.src = match.image_path;
            img.src = match.url;
            img.alt = match.dish;
            img.onerror = () => {
                img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjBmMGYwIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
            };
            const caption = document.createElement('p');
            caption.textContent = `${match.dish} (${(match.similarity * 100).toFixed(1)}%)`;
            imageWrapper.appendChild(img);
            imageWrapper.appendChild(caption);
            imagesDiv.appendChild(imageWrapper);
        });
        resultGroup.appendChild(title);
        resultGroup.appendChild(ingredientsDiv);
        resultGroup.appendChild(imagesDiv);
        resultsContainer.appendChild(resultGroup);
    });
    resultsSection.style.display = 'block';
}

// Log function
function log(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logDiv = document.createElement('div');
    logDiv.className = `log-${type}`;
    logDiv.textContent = `[${timestamp}] ${message}`;
    consoleOutput.appendChild(logDiv);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
    
    // Also log to browser console
    console.log(`[${type.toUpperCase()}] ${message}`);
}


// Assign event listeners only (no redeclaration)
document.getElementById('upload-btn').addEventListener('click', () => menuImageInput.click());
document.getElementById('camera-btn').addEventListener('click', () => cameraInput.click());
menuImageInput.addEventListener('change', handleFileSelect);
removeImageBtn.addEventListener('click', removeImage);

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            selectedImage = e.target.result;
            previewImg.src = selectedImage;
            imagePreview.style.display = 'block';
            uploadArea.style.display = 'none';
            updateProcessButton();
            log('📸 Image uploaded successfully', 'success');
        };
        reader.readAsDataURL(file);
    }
}

function removeImage() {
    selectedImage = null;
    imagePreview.style.display = 'none';
    uploadArea.style.display = 'block';
    menuImageInput.value = '';
    updateProcessButton();
    log('🗑️ Image removed', 'info');
} 