import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme.dart';
import '../../services/firebase_service.dart';

class PreCheckScreen extends StatefulWidget {
  final String category;
  final String subCategory;
  final String option;

  const PreCheckScreen({
    super.key,
    required this.category,
    required this.subCategory,
    required this.option,
  });

  @override
  State<PreCheckScreen> createState() => _PreCheckScreenState();
}

class _PreCheckScreenState extends State<PreCheckScreen> {
  bool _saved = false;

  @override
  void initState() {
    super.initState();
    FirebaseService.setActivePatient();  // tells ESP32 which patient
    FirebaseService.requestScan();       // triggers Python skin analysis
  }

  Future<void> _saveToDoctor(Map<String, dynamic> vitals) async {
    final user = FirebaseService.currentUser;
    if (user == null) return;
    final userDoc = await FirebaseFirestore.instance
        .collection('users')
        .doc(user.uid)
        .get();
    final name = userDoc.data()?['name'] ?? 'Unknown';

    await FirebaseService.savePreCheckSession(
      patientId: user.uid,
      patientName: name,
      category: widget.category,
      subCategory: widget.subCategory,
      option: widget.option,
      vitals: vitals,
      skinReadiness: '--',
    );
    setState(() => _saved = true);
  }

  @override
  Widget build(BuildContext context) {
    final patientId = FirebaseService.currentUser?.uid ?? '';

    return Scaffold(
      backgroundColor: AppTheme.lightGrey,
      appBar: AppBar(
        backgroundColor: AppTheme.darkNavy,
        iconTheme: const IconThemeData(color: Colors.white),
        title: Text('Pre-Check',
            style: GoogleFonts.poppins(
                color: Colors.white, fontWeight: FontWeight.w600)),
      ),
      body: StreamBuilder<DocumentSnapshot>(
        stream: FirebaseService.listenToVitals(patientId),
        builder: (context, snapshot) {

          String bpm  = '--';
          String spo2 = '--';
          String skinReadiness = '--';
          Map<String, dynamic>? skinDetails;
          bool esp32Online = false;

          if (snapshot.hasData && snapshot.data!.exists) {
            final data = snapshot.data!.data() as Map<String, dynamic>;
            final status = data['status'] as String? ?? '';

            esp32Online = status == 'active';

            final rawBpm  = data['bpm'];
            final rawSpo2 = data['spo2'];

            bpm  = (rawBpm  != null && rawBpm  != -1) ? '$rawBpm'  : '--';
            spo2 = (rawSpo2 != null && rawSpo2 != -1) ? '$rawSpo2' : '--';

            skinReadiness = data['skinReadiness'] as String? ?? '--';
            skinDetails   = data['skinDetails']   as Map<String, dynamic>?;
          }

          final vitalsSnapshot = {
            'heartRate'  : bpm  == '--' ? '--' : int.tryParse(bpm)  ?? '--',
            'oxygen'     : spo2 == '--' ? '--' : int.tryParse(spo2) ?? '--',
            'temperature': '--',
          };

          final isReady   = skinReadiness == 'Ready';
          final isPending = skinReadiness == '--';

          return SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 10),

                // ── Operation Summary ────────────────────────────────────
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: AppTheme.darkNavy,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Selected Procedure',
                          style: GoogleFonts.poppins(
                              color: Colors.white60, fontSize: 12)),
                      const SizedBox(height: 4),
                      Text(widget.category,
                          style: GoogleFonts.poppins(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.w700)),
                      Text(
                          '${widget.subCategory.toUpperCase()} → ${widget.option}',
                          style: GoogleFonts.poppins(
                              color: AppTheme.accent, fontSize: 13)),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // ── Vitals ───────────────────────────────────────────────
                Text('Live Vitals',
                    style: GoogleFonts.poppins(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.darkNavy)),
                Text(
                  esp32Online
                      ? 'Sensor connected — reading live'
                      : 'Place finger on MAX30105 sensor',
                  style: GoogleFonts.poppins(
                      fontSize: 13, color: AppTheme.textGrey),
                ),
                const SizedBox(height: 16),

                GridView.count(
                  crossAxisCount: 2,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  mainAxisSpacing: 16,
                  crossAxisSpacing: 16,
                  childAspectRatio: 1.4,
                  children: [
                    _VitalCard(
                      label: 'Oxygen (SpO₂)',
                      value: spo2 == '--' ? '--' : '$spo2%',
                      icon: Icons.air,
                      color: Colors.blue,
                      isOffline: spo2 == '--',
                    ),
                    _VitalCard(
                      label: 'Heart Rate',
                      value: bpm == '--' ? '--' : '$bpm bpm',
                      icon: Icons.favorite,
                      color: Colors.red,
                      isOffline: bpm == '--',
                    ),
                  ],
                ),

                // ── Sensor offline notice ────────────────────────────────
                if (!esp32Online) ...[
                  const SizedBox(height: 12),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 10),
                    decoration: BoxDecoration(
                      color: Colors.orange.shade50,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.orange.shade200),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.sensors_off,
                            color: Colors.orange.shade400, size: 18),
                        const SizedBox(width: 10),
                        Text('Sensor offline — place finger on MAX30105',
                            style: GoogleFonts.poppins(
                                fontSize: 12,
                                color: Colors.orange.shade700)),
                      ],
                    ),
                  ),
                ],

                const SizedBox(height: 24),

                // ── Skin Readiness ───────────────────────────────────────
                Text('Skin Readiness',
                    style: GoogleFonts.poppins(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.darkNavy)),
                Text('Analysed by AI camera system',
                    style: GoogleFonts.poppins(
                        fontSize: 13, color: AppTheme.textGrey)),
                const SizedBox(height: 12),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: isPending
                        ? Colors.grey.shade100
                        : isReady
                            ? Colors.green.shade50
                            : Colors.red.shade50,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(
                      color: isPending
                          ? Colors.grey.shade300
                          : isReady
                              ? Colors.green.shade300
                              : Colors.red.shade300,
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            isPending
                                ? Icons.hourglass_empty
                                : isReady
                                    ? Icons.check_circle
                                    : Icons.cancel,
                            color: isPending
                                ? Colors.grey
                                : isReady
                                    ? Colors.green
                                    : Colors.red,
                            size: 28,
                          ),
                          const SizedBox(width: 14),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                isPending ? 'Awaiting scan...' : skinReadiness,
                                style: GoogleFonts.poppins(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w700,
                                  color: isPending
                                      ? Colors.grey
                                      : isReady
                                          ? Colors.green
                                          : Colors.red,
                                ),
                              ),
                              if (!isPending)
                                Text(
                                  isReady
                                      ? 'No severe lesions detected'
                                      : 'Surgery not recommended',
                                  style: GoogleFonts.poppins(
                                      fontSize: 12,
                                      color: AppTheme.textGrey),
                                ),
                            ],
                          ),
                        ],
                      ),
                      if (skinDetails != null) ...[
                        const SizedBox(height: 10),
                        const Divider(),
                        const SizedBox(height: 6),
                        Text(
                          'Lesions: ${skinDetails['severeLesionCount'] ?? '--'}',
                          style: GoogleFonts.poppins(
                              fontSize: 12, color: AppTheme.textGrey),
                        ),
                        Text(
                          'Coverage: ${skinDetails['coveragePercent'] ?? '--'}%',
                          style: GoogleFonts.poppins(
                              fontSize: 12, color: AppTheme.textGrey),
                        ),
                        Text(
                          '${skinDetails['reason'] ?? ''}',
                          style: GoogleFonts.poppins(
                              fontSize: 12, color: AppTheme.textGrey),
                        ),
                      ],
                    ],
                  ),
                ),
                const SizedBox(height: 30),

                // ── Send to Doctor Button ────────────────────────────────
                if (_saved)
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.green.shade50,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.green.shade200),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.check_circle, color: Colors.green),
                        const SizedBox(width: 8),
                        Text('Sent to Doctor Dashboard',
                            style: GoogleFonts.poppins(
                                color: Colors.green,
                                fontWeight: FontWeight.w600)),
                      ],
                    ),
                  )
                else
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: () => _saveToDoctor(vitalsSnapshot),
                      icon: const Icon(Icons.send),
                      label: const Text('Send to Doctor'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.blue,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14)),
                      ),
                    ),
                  ),
                const SizedBox(height: 16),
              ],
            ),
          );
        },
      ),
    );
  }
}

class _VitalCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color color;
  final bool isOffline;

  const _VitalCard({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
    this.isOffline = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: isOffline ? Colors.grey.shade400 : color, size: 24),
          const SizedBox(height: 8),
          Text(value,
              style: GoogleFonts.poppins(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: isOffline ? Colors.grey.shade400 : AppTheme.darkNavy)),
          Text(label,
              style: GoogleFonts.poppins(fontSize: 11, color: AppTheme.textGrey)),
        ],
      ),
    );
  }
}