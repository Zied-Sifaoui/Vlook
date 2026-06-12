import 'dart:convert';
import 'package:http/http.dart' as http;

class EspService {
  static const String espIp = '10.161.145.79';
  static const String baseUrl = 'http://$espIp';

  // ── All methods silently fail if ESP32 is not connected ───────────────
  // No errors or crashes — just returns false or null

  static Future<bool> sendMode(int mode) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/set?mode=$mode'))
          .timeout(const Duration(seconds: 3));
      return response.statusCode == 200;
    } catch (_) {
      return false; // ESP32 offline — ignore silently
    }
  }

  static Future<bool> sendColor(int colorIdx) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/set?color=$colorIdx'))
          .timeout(const Duration(seconds: 3));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> reset() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/reset'))
          .timeout(const Duration(seconds: 3));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<Map<String, dynamic>?> getStatus() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/status'))
          .timeout(const Duration(seconds: 3));
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      return null;
    } catch (_) {
      return null;
    }
  }

  static Future<bool> sendFaceSurgery(String option, bool isDeaf) async {
    int mode;
    if (isDeaf) {
      mode = 6;
    } else {
      switch (option) {
        case 'nose':
          mode = 4;
          break;
        case 'scar':
          mode = 2;
          break;
        case 'mouth':
          mode = 3;
          break;
        default:
          mode = 2;
      }
    }
    return await sendMode(mode);
  }

  static Future<bool> sendHair(String subCategory, bool isDeaf) async {
    int mode;
    if (isDeaf) {
      mode = 5;
    } else {
      switch (subCategory) {
        case 'hair':
          mode = 0;
          break;
        case 'eyebrows':
          mode = 1;
          break;
        default:
          mode = 0;
      }
    }
    return await sendMode(mode);
  }
}