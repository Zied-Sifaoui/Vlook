import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme.dart';
import '../../services/firebase_service.dart';
import '../login_screen.dart';
import 'face_surgery_screen.dart';
import 'hair_screen.dart';

class PatientMenuScreen extends StatefulWidget {
  const PatientMenuScreen({super.key});

  @override
  State<PatientMenuScreen> createState() => _PatientMenuScreenState();
}

class _PatientMenuScreenState extends State<PatientMenuScreen> {
  bool _isDeaf = false;
  bool _beforeAfter = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.lightGrey,
      appBar: AppBar(
        backgroundColor: AppTheme.darkNavy,
        title: Text('MedAR',
            style: GoogleFonts.montserrat(
                color: Colors.white, fontWeight: FontWeight.w800)),
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
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 20),
            Text('Choose Operation',
                style: GoogleFonts.poppins(
                    fontSize: 24,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.darkNavy)),
            Text('Select the procedure you want to visualize',
                style: GoogleFonts.poppins(
                    fontSize: 14, color: AppTheme.textGrey)),
            const SizedBox(height: 24),

            // ── Sign Language Toggle ─────────────────────────────────
            _ToggleOption(
              emoji: '🤟',
              title: 'Sign Language Mode',
              subtitle: 'Enable if you are deaf',
              value: _isDeaf,
              activeColor: AppTheme.accent,
              onChanged: (val) => setState(() => _isDeaf = val),
            ),

            if (_isDeaf) ...[
              const SizedBox(height: 8),
              _InfoBanner(
                color: AppTheme.accent,
                text: 'Sign language detection will be activated on the Jetson screen',
              ),
            ],

            const SizedBox(height: 12),

            // ── Before / After Toggle ────────────────────────────────
            _ToggleOption(
              emoji: '🔄',
              title: 'Before / After Review',
              subtitle: 'Show comparison on the screen',
              value: _beforeAfter,
              activeColor: const Color(0xFF6A1B9A),
              onChanged: (val) => setState(() => _beforeAfter = val),
            ),

            if (_beforeAfter) ...[
              const SizedBox(height: 8),
              _InfoBanner(
                color: const Color(0xFF6A1B9A),
                text: 'Before/After comparison will be displayed on the Jetson screen',
              ),
            ],

            const SizedBox(height: 24),

            // ── Operation Cards ──────────────────────────────────────
            _OperationCard(
              icon: Icons.face,
              title: 'Face Surgery',
              subtitle: 'Nose • Scar • Mouth',
              color: AppTheme.blue,
              isDeaf: _isDeaf,
              beforeAfter: _beforeAfter,
              onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (_) => FaceSurgeryScreen(
                            isDeaf: _isDeaf,
                            beforeAfter: _beforeAfter,
                          ))),
            ),
            const SizedBox(height: 20),
            _OperationCard(
              icon: Icons.self_improvement,
              title: 'Hair Implementation',
              subtitle: 'Hair Styles • Eyebrows',
              color: AppTheme.accent,
              isDeaf: _isDeaf,
              beforeAfter: _beforeAfter,
              onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (_) => HairScreen(
                            isDeaf: _isDeaf,
                            beforeAfter: _beforeAfter,
                          ))),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Reusable Toggle Option ────────────────────────────────────────

class _ToggleOption extends StatelessWidget {
  final String emoji;
  final String title;
  final String subtitle;
  final bool value;
  final Color activeColor;
  final ValueChanged<bool> onChanged;

  const _ToggleOption({
    required this.emoji,
    required this.title,
    required this.subtitle,
    required this.value,
    required this.activeColor,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: value ? activeColor.withAlpha(25) : Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: value ? activeColor : Colors.grey.shade200,
          width: 2,
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: value ? activeColor.withAlpha(50) : Colors.grey.shade100,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(emoji, style: const TextStyle(fontSize: 24)),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: GoogleFonts.poppins(
                        fontWeight: FontWeight.w600,
                        fontSize: 15,
                        color: AppTheme.darkNavy)),
                Text(subtitle,
                    style: GoogleFonts.poppins(
                        fontSize: 12, color: AppTheme.textGrey)),
              ],
            ),
          ),
          Switch(
            value: value,
            onChanged: onChanged,
            activeColor: activeColor,
          ),
        ],
      ),
    );
  }
}

// ── Reusable Info Banner ──────────────────────────────────────────

class _InfoBanner extends StatelessWidget {
  final Color color;
  final String text;

  const _InfoBanner({required this.color, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: color.withAlpha(20),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          Icon(Icons.info_outline, color: color, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(text,
                style: GoogleFonts.poppins(fontSize: 12, color: color)),
          ),
        ],
      ),
    );
  }
}

// ── Operation Card ────────────────────────────────────────────────

class _OperationCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Color color;
  final bool isDeaf;
  final bool beforeAfter;
  final VoidCallback onTap;

  const _OperationCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.color,
    required this.isDeaf,
    required this.beforeAfter,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(28),
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: color.withAlpha(89),
              blurRadius: 20,
              offset: const Offset(0, 8),
            )
          ],
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.white.withAlpha(50),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: Colors.white, size: 32),
            ),
            const SizedBox(width: 20),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(title,
                          style: GoogleFonts.poppins(
                              fontSize: 20,
                              fontWeight: FontWeight.w700,
                              color: Colors.white)),
                      if (isDeaf) ...[
                        const SizedBox(width: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.white.withAlpha(64),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: const Text('🤟',
                              style: TextStyle(fontSize: 13)),
                        ),
                      ],
                      if (beforeAfter) ...[
                        const SizedBox(width: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.white.withAlpha(64),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: const Text('🔄',
                              style: TextStyle(fontSize: 13)),
                        ),
                      ],
                    ],
                  ),
                  Text(subtitle,
                      style: GoogleFonts.poppins(
                          fontSize: 13,
                          color: Colors.white.withAlpha(204))),
                ],
              ),
            ),
            const Icon(Icons.arrow_forward_ios,
                color: Colors.white, size: 18),
          ],
        ),
      ),
    );
  }
}