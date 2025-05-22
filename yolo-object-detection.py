from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import logging
import time
import os
import threading
from flask_cors import CORS  # Ajout de CORS pour les requêtes cross-origin

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration YOLOv4
MODEL_CONFIDENCE = 0.4  # Seuil de confiance plus bas pour plus de rapidité
NMS_THRESHOLD = 0.4     # Seuil NMS plus élevé pour moins de chevauchements

# Cache pour les résultats récents
detection_cache = {}
cache_lock = threading.Lock()
MAX_CACHE_SIZE = 10
CACHE_TTL = 3  # secondes

# Charger le modèle YOLO - utiliser une version plus légère
try:
    # Utiliser YOLOv4-tiny pour plus de rapidité
    if os.path.exists('yolov4-tiny.weights') and os.path.exists('yolov4-tiny.cfg'):
        logger.info("Chargement du modèle YOLOv4-tiny...")
        net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
    else:
        logger.info("Modèle tiny non trouvé, chargement du modèle YOLOv4 standard...")
        net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
    
    # Optimiser le backend
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    
    # Utiliser CUDA si disponible
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        logger.info("CUDA détecté, utilisation de GPU...")
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    ln = net.getLayerNames()
    try:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Chargement des classes
    LABELS = open('coco.names').read().strip().split("\n")
    logger.info(f"Modèle YOLO chargé avec succès: {len(LABELS)} classes disponibles")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle YOLO: {e}")
    net = None
    LABELS = []

def preprocess_image(image_data, target_size=(320, 320)):
    """Prétraite l'image pour accélérer le traitement"""
    try:
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Redimensionner l'image pour un traitement plus rapide
        original_size = image.shape[:2]  # (height, width)
        image_resized = cv2.resize(image, target_size)
        
        return image, image_resized, original_size
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement de l'image: {e}")
        return None, None, None

def generate_cache_key(image_data):
    """Génère une clé de cache basée sur un échantillon de l'image"""
    # Utiliser un sous-échantillon de l'image comme clé de cache
    sample = image_data[::100]  # Prendre 1 byte sur 100
    return hash(bytes(sample))

def get_cached_detection(key):
    """Récupère une détection mise en cache si elle existe et est récente"""
    with cache_lock:
        if key in detection_cache:
            timestamp, data = detection_cache[key]
            if time.time() - timestamp < CACHE_TTL:
                return data
    return None

def set_cached_detection(key, data):
    """Met en cache une détection"""
    with cache_lock:
        # Supprimer les anciennes entrées si nécessaire
        if len(detection_cache) >= MAX_CACHE_SIZE:
            oldest_key = min(detection_cache.keys(), key=lambda k: detection_cache[k][0])
            del detection_cache[oldest_key]
        
        detection_cache[key] = (time.time(), data)

@app.route('/', methods=['POST'])
def detect():
    start_time = time.time()
    logger.info("=== Nouvelle requête reçue ===")
    
    if net is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Vérification du contenu de la requête
        if not request.is_json:
            logger.error("La requête n'est pas au format JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'image' not in data:
            logger.error("Aucune clé 'image' trouvée dans la requête JSON")
            return jsonify({'error': 'No image key found in request'}), 400
        
        # Décodage de l'image
        try:
            image_data = base64.b64decode(data['image'])
            logger.info(f"Image décodée, taille: {len(image_data)} bytes")
            
            # Vérifier le cache avant de traiter
            cache_key = generate_cache_key(image_data)
            cached_result = get_cached_detection(cache_key)
            if cached_result:
                logger.info("Résultat trouvé dans le cache")
                cached_result['processing_time']['total'] = f"{time.time() - start_time:.2f}"
                cached_result['from_cache'] = True
                return jsonify(cached_result)
            
        except Exception as e:
            logger.error(f"Erreur lors du décodage base64: {e}")
            return jsonify({'error': 'Invalid base64 encoding'}), 400
        
        # Prétraitement de l'image
        image, image_resized, original_size = preprocess_image(image_data)
        if image is None:
            return jsonify({'error': 'Failed to process image'}), 400
            
        # Détection d'objets avec l'image redimensionnée
        objects_detected = []
        detection_boxes = []
        
        try:
            H, W = original_size
            
            # Utiliser une taille d'entrée plus petite pour plus de rapidité
            blob = cv2.dnn.blobFromImage(image_resized, 1/255, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            
            # Mesure du temps de traitement du réseau
            inference_start = time.time()
            layerOutputs = net.forward(ln)
            inference_time = (time.time() - inference_start) * 1000  # en ms
            logger.info(f"Temps d'inférence: {inference_time:.2f} ms")
            
            boxes = []
            confidences = []
            classIDs = []
            
            # Facteurs d'échelle pour reconvertir les coordonnées
            h_scale = H / image_resized.shape[0]
            w_scale = W / image_resized.shape[1]
            
            # Parcours des résultats de la détection
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    
                    # Filtrer selon le seuil de confiance
                    if confidence > MODEL_CONFIDENCE:
                        box = detection[0:4] * np.array([image_resized.shape[1], image_resized.shape[0], 
                                                         image_resized.shape[1], image_resized.shape[0]])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        
                        # Reconvertir à l'échelle originale
                        x = int(x * w_scale)
                        y = int(y * h_scale)
                        width = int(width * w_scale)
                        height = int(height * h_scale)
                        
                        boxes.append([x, y, width, height])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            
            # Appliquer NMS pour éliminer les détections redondantes
            if boxes:
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, MODEL_CONFIDENCE, NMS_THRESHOLD)
                
                if len(idxs) > 0:
                    # Limiter le nombre d'objets à afficher pour de meilleures performances
                    MAX_OBJECTS = 10
                    idxs = idxs.flatten()[:MAX_OBJECTS]
                    
                    for i in idxs:
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        
                        # Traitement pour l'affichage sur l'image
                        color = (0, 255, 0)
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Ajouter à la liste des objets détectés
                        objects_detected.append(LABELS[classIDs[i]])
                        
                        # Ajouter les informations de cadre normalisées (pour l'affichage côté client)
                        detection_boxes.append({
                            'x': float(x) / W,
                            'y': float(y) / H,
                            'width': float(w) / W,
                            'height': float(h) / H,
                            'label': LABELS[classIDs[i]],
                            'confidence': float(confidences[i])
                        })
            
            logger.info(f"Objets détectés: {objects_detected}")
        except Exception as e:
            logger.error(f"Erreur pendant la détection d'objets: {e}")
            return jsonify({'error': f'Detection error: {str(e)}'}), 500
        
        # Encodage de l'image résultante avec compression
        try:
            # Utiliser un facteur de qualité plus bas pour réduire la taille
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage de l'image résultante: {e}")
            return jsonify({'error': f'Image encoding error: {str(e)}'}), 500
        
        # Création de la réponse
        response = {
            'image': img_base64, 
            'objects': list(set(objects_detected)),  # Enlever les doublons
            'boxes': detection_boxes,  # Informations sur les cadres de détection
            'processing_time': {
                'total': f"{time.time() - start_time:.2f}",
                'inference': f"{inference_time:.2f}"
            },
            'from_cache': False
        }
        
        # Mettre en cache le résultat
        set_cached_detection(cache_key, response)
        
        # Journaliser des informations sur le temps de traitement
        total_time = time.time() - start_time
        logger.info(f"Réponse envoyée avec {len(objects_detected)} objets en {total_time:.2f}s")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erreur générale non gérée: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'ok', 
        'message': 'Server is running',
        'model_loaded': net is not None,
        'classes_available': len(LABELS),
        'model_type': 'YOLOv4-tiny' if 'tiny' in str(net) else 'YOLOv4'
    }), 200

@app.route('/config', methods=['GET', 'POST'])
def config():
    global MODEL_CONFIDENCE, NMS_THRESHOLD, CACHE_TTL
    
    if request.method == 'POST':
        data = request.get_json()
        
        if 'confidence' in data:
            MODEL_CONFIDENCE = float(data['confidence'])
        
        if 'nms_threshold' in data:
            NMS_THRESHOLD = float(data['nms_threshold'])
        
        if 'cache_ttl' in data:
            CACHE_TTL = int(data['cache_ttl'])
            
        return jsonify({'status': 'ok', 'message': 'Configuration updated'})
    else:
        return jsonify({
            'confidence': MODEL_CONFIDENCE,
            'nms_threshold': NMS_THRESHOLD,
            'cache_ttl': CACHE_TTL
        })

if __name__ == '__main__':
    logger.info("Démarrage du serveur Flask sur le port 5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)