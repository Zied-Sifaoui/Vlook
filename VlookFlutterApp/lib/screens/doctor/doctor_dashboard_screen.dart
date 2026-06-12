import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme.dart';
import '../../services/firebase_service.dart';
import '../login_screen.dart';

class DoctorDashboardScreen extends StatefulWidget {
  const DoctorDashboardScreen({super.key});

  @override
  State<DoctorDashboardScreen> createState() => _DoctorDashboardScreenState();
}

class _DoctorDashboardScreenState extends State<DoctorDashboardScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.lightGrey,
      appBar: AppBar(
        backgroundColor: AppTheme.darkNavy,
        automaticallyImplyLeading: false,
        title: Text('Doctor Dashboard',
            style: GoogleFonts.montserrat(
                color: Colors.white, fontWeight: FontWeight.w700)),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout, color: Colors.white),
            onPressed: () async {
              await FirebaseService.signOut();
              if (context.mounted) {
                Navigator.pushReplacement(context,
                    MaterialPageRoute(builder: (_) => const LoginScreen()));
              }
            },
          )
        ],
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: AppTheme.accent,
          labelStyle: GoogleFonts.poppins(fontWeight: FontWeight.w600),
          tabs: const [
            Tab(text: 'Sessions', icon: Icon(Icons.history)),
            Tab(text: 'Patients', icon: Icon(Icons.people)),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: const [
          _SessionsTab(),
          _PatientsTab(),
        ],
      ),
    );
  }
}

// ── Sessions Tab ──────────────────────────────────────────────────────────────

class _SessionsTab extends StatelessWidget {
  const _SessionsTab();

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<QuerySnapshot>(
      stream: FirebaseService.streamAllSessions(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }
        if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.inbox, size: 60, color: Colors.grey.shade300),
                const SizedBox(height: 12),
                Text('No sessions yet',
                    style: GoogleFonts.poppins(color: AppTheme.textGrey)),
              ],
            ),
          );
        }

        final docs = snapshot.data!.docs;
        return ListView.builder(
          padding: const EdgeInsets.all(16),
          itemCount: docs.length,
          itemBuilder: (context, i) {
            final data = docs[i].data() as Map<String, dynamic>;
            final vitals = data['vitals'] as Map<String, dynamic>? ?? {};
            final ts = data['timestamp'] as Timestamp?;
            final date = ts != null
                ? '${ts.toDate().day}/${ts.toDate().month}/${ts.toDate().year}  ${ts.toDate().hour}:${ts.toDate().minute.toString().padLeft(2, '0')}'
                : 'Unknown';
            final patientId = data['patientId'] as String? ?? '';

            return Container(
              margin: const EdgeInsets.only(bottom: 14),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 10,
                    offset: const Offset(0, 3),
                  )
                ],
              ),
              child: ExpansionTile(
                leading: CircleAvatar(
                  backgroundColor: AppTheme.blue.withOpacity(0.1),
                  child: Text(
                    (data['patientName'] as String? ?? '?')[0].toUpperCase(),
                    style: GoogleFonts.poppins(
                        color: AppTheme.blue, fontWeight: FontWeight.w700),
                  ),
                ),
                title: Text(data['patientName'] ?? 'Unknown',
                    style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
                subtitle: Text(
                    '${data['category']} → ${data['subCategory']} → ${data['option']}',
                    style: GoogleFonts.poppins(
                        fontSize: 12, color: AppTheme.textGrey)),
                trailing: Text(date,
                    style: GoogleFonts.poppins(
                        fontSize: 11, color: AppTheme.textGrey)),
                children: [
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // ── Vitals Row ──────────────────────────────────────
                        Row(
                          children: [
                            Expanded(
                              child: _VitalTile(
                                  label: 'SpO₂',
                                  value: '${vitals['oxygen'] ?? '--'}%',
                                  color: Colors.blue),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: _VitalTile(
                                  label: 'Heart Rate',
                                  value: '${vitals['heartRate'] ?? '--'} bpm',
                                  color: Colors.red),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: _VitalTile(
                                  label: 'Temp',
                                  value: '${vitals['temperature'] ?? '--'}°C',
                                  color: Colors.orange),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),

                        // ── Skin Readiness (live from vitals) ───────────────
                        StreamBuilder<DocumentSnapshot>(
                          stream: FirebaseService.streamPatientVitals(patientId),
                          builder: (context, vitalsSnap) {
                            // Fall back to session value, override with live vitals
                            String skinReadiness =
                                data['skinReadiness'] as String? ?? '--';
                            Map<String, dynamic>? skinDetails;

                            if (vitalsSnap.hasData && vitalsSnap.data!.exists) {
                              final vData = vitalsSnap.data!.data()
                                  as Map<String, dynamic>;
                              final live = vData['skinReadiness'] as String?;
                              if (live != null && live.isNotEmpty) {
                                skinReadiness = live;
                              }
                              skinDetails = vData['skinDetails']
                                  as Map<String, dynamic>?;
                            }

                            final isReady = skinReadiness == 'Ready';
                            final isPending = skinReadiness == '--';

                            return Container(
                              width: double.infinity,
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 16, vertical: 12),
                              decoration: BoxDecoration(
                                color: isPending
                                    ? Colors.grey.shade100
                                    : isReady
                                        ? Colors.green.shade50
                                        : Colors.red.shade50,
                                borderRadius: BorderRadius.circular(12),
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
                                        size: 22,
                                      ),
                                      const SizedBox(width: 10),
                                      Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Text('Skin Readiness',
                                              style: GoogleFonts.poppins(
                                                  fontSize: 11,
                                                  color: AppTheme.textGrey)),
                                          Text(
                                            isPending
                                                ? 'Pending'
                                                : skinReadiness,
                                            style: GoogleFonts.poppins(
                                              fontSize: 15,
                                              fontWeight: FontWeight.w700,
                                              color: isPending
                                                  ? Colors.grey
                                                  : isReady
                                                      ? Colors.green
                                                      : Colors.red,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ],
                                  ),
                                  // ── Extra details from Python ─────────────
                                  if (skinDetails != null) ...[
                                    const SizedBox(height: 8),
                                    Text(
                                      'Lesions: ${skinDetails['severeLesionCount'] ?? '--'}   '
                                      'Coverage: ${skinDetails['coveragePercent'] ?? '--'}%   '
                                      '${skinDetails['reason'] ?? ''}',
                                      style: GoogleFonts.poppins(
                                          fontSize: 11,
                                          color: AppTheme.textGrey),
                                    ),
                                  ],
                                ],
                              ),
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }
}

class _VitalTile extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _VitalTile(
      {required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(label,
                    style: GoogleFonts.poppins(
                        fontSize: 10, color: AppTheme.textGrey)),
                Text(value,
                    style: GoogleFonts.poppins(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: AppTheme.darkNavy)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ── Patients Tab ──────────────────────────────────────────────────────────────

class _PatientsTab extends StatelessWidget {
  const _PatientsTab();

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<QuerySnapshot>(
      stream: FirebaseService.streamAllPatients(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }
        if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
          return Center(
            child: Text('No patients registered yet',
                style: GoogleFonts.poppins(color: AppTheme.textGrey)),
          );
        }

        final docs = snapshot.data!.docs;
        return ListView.builder(
          padding: const EdgeInsets.all(16),
          itemCount: docs.length,
          itemBuilder: (context, i) {
            final data = docs[i].data() as Map<String, dynamic>;
            return Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.04),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  )
                ],
              ),
              child: Row(
                children: [
                  CircleAvatar(
                    backgroundColor: AppTheme.lightBlue,
                    child: Text(
                      (data['name'] as String? ?? '?')[0].toUpperCase(),
                      style: const TextStyle(
                          color: Colors.white, fontWeight: FontWeight.bold),
                    ),
                  ),
                  const SizedBox(width: 14),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(data['name'] ?? 'Unknown',
                          style: GoogleFonts.poppins(
                              fontWeight: FontWeight.w600,
                              color: AppTheme.darkNavy)),
                      Text(data['email'] ?? '',
                          style: GoogleFonts.poppins(
                              fontSize: 12, color: AppTheme.textGrey)),
                    ],
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }
}