import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class FirebaseService {
  static final FirebaseAuth _auth = FirebaseAuth.instance;
  static final FirebaseFirestore _db = FirebaseFirestore.instance;

  // ─── AUTH ────────────────────────────────────────────────────────────────────

  static Future<UserCredential?> loginAsDoctor(
      String email, String password) async {
    try {
      final cred = await _auth.signInWithEmailAndPassword(
          email: email, password: password);
      final doc = await _db.collection('users').doc(cred.user!.uid).get();
      if (doc.data()?['role'] != 'doctor') {
        await _auth.signOut();
        throw Exception('Not a doctor account');
      }
      return cred;
    } catch (e) {
      rethrow;
    }
  }

  static Future<UserCredential?> loginAsPatient(
      String email, String password) async {
    try {
      final cred = await _auth.signInWithEmailAndPassword(
          email: email, password: password);
      final doc = await _db.collection('users').doc(cred.user!.uid).get();
      if (doc.data()?['role'] != 'patient') {
        await _auth.signOut();
        throw Exception('Not a patient account');
      }
      return cred;
    } catch (e) {
      rethrow;
    }
  }

  static Future<void> registerPatient(
      String email, String password, String name) async {
    final cred = await _auth.createUserWithEmailAndPassword(
        email: email, password: password);
    await _db.collection('users').doc(cred.user!.uid).set({
      'name': name,
      'email': email,
      'role': 'patient',
      'createdAt': FieldValue.serverTimestamp(),
    });
  }

  static Future<void> registerDoctor(
      String email, String password, String name) async {
    final cred = await _auth.createUserWithEmailAndPassword(
        email: email, password: password);
    await _db.collection('users').doc(cred.user!.uid).set({
      'name': name,
      'email': email,
      'role': 'doctor',
      'createdAt': FieldValue.serverTimestamp(),
    });
  }

  static Future<void> signOut() => _auth.signOut();

  static User? get currentUser => _auth.currentUser;

  // ─── FILTER CONTROL ──────────────────────────────────────────────────────────

  static Future<void> sendFilterToJetson({
    required String category,
    required String subCategory,
    required String option,
    bool beforeAfter = false,
  }) async {
    await _db.collection('jetson_control').doc('active_filter').set({
      'category': category,
      'subCategory': subCategory,
      'option': option,
      'beforeAfter': beforeAfter,
      'updatedAt': FieldValue.serverTimestamp(),
      'patientId': _auth.currentUser?.uid,
    });
  }

  // ─── ACTIVE PATIENT ───────────────────────────────────────────────────────────

  static Future<void> setActivePatient() async {
    final uid = _auth.currentUser?.uid;
    if (uid == null) return;
    await _db.collection('jetson_control').doc('active_filter').set(
      {'patientId': uid, 'updatedAt': FieldValue.serverTimestamp()},
      SetOptions(merge: true),
    );
  }

  // ─── SCAN TRIGGER ────────────────────────────────────────────────────────────

  static Future<void> requestScan() async {
    await _db.collection('jetson_control').doc('active_filter').update({
      'scanRequested': true,
      'scanRequestedAt': FieldValue.serverTimestamp(),
    });
  }

  // ─── VITALS ──────────────────────────────────────────────────────────────────

  static Stream<DocumentSnapshot> listenToVitals(String patientId) {
    return _db.collection('vitals').doc(patientId).snapshots();
  }

  static Stream<DocumentSnapshot> streamPatientVitals(String patientId) {
    return _db.collection('vitals').doc(patientId).snapshots();
  }

  // ─── PRE-CHECK SESSION ────────────────────────────────────────────────────────

  static Future<void> savePreCheckSession({
    required String patientId,
    required String patientName,
    required String category,
    required String subCategory,
    required String option,
    required Map<String, dynamic> vitals,
    required String skinReadiness,
  }) async {
    await _db.collection('sessions').add({
      'patientId': patientId,
      'patientName': patientName,
      'category': category,
      'subCategory': subCategory,
      'option': option,
      'vitals': vitals,
      'skinReadiness': skinReadiness,
      'timestamp': FieldValue.serverTimestamp(),
    });
  }

  // ─── DOCTOR DASHBOARD ─────────────────────────────────────────────────────────

  static Stream<QuerySnapshot> streamAllSessions() {
    return _db
        .collection('sessions')
        .orderBy('timestamp', descending: true)
        .snapshots();
  }

  static Stream<QuerySnapshot> streamAllPatients() {
    return _db
        .collection('users')
        .where('role', isEqualTo: 'patient')
        .snapshots();
  }
}