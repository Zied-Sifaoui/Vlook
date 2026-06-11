
# Vlook System — Project Summary

## English
Vlook is an AR-based pre-operative medical visualization system for cosmetic surgery. It integrates a Flutter mobile app (patient + doctor dashboard), an ESP32 health-check module (heart rate, SpO₂, temperature via MAX30102/MLX90614), and a Jetson Nano running MediaPipe (468 facial landmarks), OpenCV (AR overlay at 28–30 fps), U-Net (acne/scar segmentation), gesture recognition, and speech-to-text. Health data and session records are synced in real time via Firebase Firestore. A hybrid wired/wireless architecture (USB, HDMI, Wi-Fi HTTP, Firebase HTTPS) connects all components. The mobile app follows the MVC pattern with FirebaseService and EspService controllers.

## Français
Vlook est un système médical de visualisation préopératoire basé sur la RA pour la chirurgie esthétique. Il combine une application mobile Flutter (patient + tableau de bord médecin), un module de contrôle santé ESP32 (fréquence cardiaque, SpO₂, température via MAX30102/MLX90614), et un Jetson Nano exécutant MediaPipe (468 repères faciaux), OpenCV (superposition RA à 28–30 ips), U-Net (segmentation acné/cicatrices), reconnaissance gestuelle, et synthèse vocale. Les données et sessions sont synchronisées en temps réel via Firebase Firestore. Une architecture hybride filaire/sans fil (USB, HDMI, Wi-Fi HTTP, Firebase HTTPS) connecte l'ensemble. L'application mobile suit le modèle MVC avec les contrôleurs FirebaseService et EspService.

## العربية
Vlook هو نظام تصوير طبي قبل العمليات الجراحية التجميلية يعتمد على الواقع المعزز. يدمج النظام تطبيق جوال Flutter (مريض + لوحة تحكم الطبيب)، وحدة فحص صحي ESP32 (معدل ضربات القلب، تشبع الأكسجين، درجة الحرارة عبر MAX30102/MLX90614)، و Jetson Nano يشغل MediaPipe (468 نقطة وجه)، OpenCV (تراكب الواقع المعزز بـ 28–30 إطار/ثانية)، U-Net (تجزئة حب الشباب والندوب)، التعرف على الإيماءات، وتحويل الكلام إلى نص. تُزامن بيانات الجلسات فورياً عبر Firebase Firestore. تربط جميع المكونات بنية هجينة سلكية/لاسلكية (USB، HDMI، Wi-Fi HTTP، Firebase HTTPS). يتبع تطبيق الجوال نمط MVC مع وحدتي FirebaseService و EspService.
