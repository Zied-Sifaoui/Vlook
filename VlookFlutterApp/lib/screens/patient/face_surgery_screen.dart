import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme.dart';
import '../../services/firebase_service.dart';
import '../../services/esp_service.dart';
import 'pre_check_screen.dart';

class FaceSurgeryScreen extends StatefulWidget {
  final bool isDeaf;
  final bool beforeAfter;
  const FaceSurgeryScreen({
    super.key,
    required this.isDeaf,
    required this.beforeAfter,
  });

  @override
  State<FaceSurgeryScreen> createState() => _FaceSurgeryScreenState();
}

class _FaceSurgeryScreenState extends State<FaceSurgeryScreen> {
  String? _selectedOption;
  String? _selectedMouthOption;
  bool _isSending = false;
  String? _espStatus;

  final List<Map<String, dynamic>> _options = [
    {
      'key': 'nose',
      'label': 'Nose',
      'icon': Icons.face_retouching_natural,
      'desc': 'Rhinoplasty simulation',
    },
    {
      'key': 'scar',
      'label': 'Scar',
      'icon': Icons.healing,
      'desc': 'Scar removal preview',
    },
    {
      'key': 'mouth',
      'label': 'Mouth',
      'icon': Icons.sentiment_satisfied_alt,
      'desc': 'Lip & smile adjustment',
    },
  ];

  final List<Map<String, dynamic>> _mouthOptions = [
    {
      'key': 'mouth_full',
      'label': 'Whole Mouth',
      'icon': Icons.crop_free,
      'desc': 'Adjustment for the entire mouth area',
    },
    {
      'key': 'mouth_upper',
      'label': 'Upper Lip',
      'icon': Icons.vertical_align_top,
      'desc': 'Adjustment for the upper lip only',
    },
    {
      'key': 'mouth_lower',
      'label': 'Lower Lip',
      'icon': Icons.vertical_align_bottom,
      'desc': 'Adjustment for the lower lip only',
    },
  ];

  Future<void> _confirm() async {
    final effectiveOption =
        _selectedOption == 'mouth' ? _selectedMouthOption : _selectedOption;
    if (effectiveOption == null) return;

    setState(() {
      _isSending = true;
      _espStatus = null;
    });

    // Build option string — includes beforeAfter flag if enabled
    final optionValue = widget.isDeaf
        ? 'sign_language'
        : widget.beforeAfter
            ? '${effectiveOption}_before_after'
            : effectiveOption;

    // Send to Firebase / Jetson
    await FirebaseService.sendFilterToJetson(
      category: 'face_surgery',
      subCategory: effectiveOption,
      option: optionValue,
      beforeAfter: widget.beforeAfter,
    );

    // Trigger the Python scan
    await FirebaseService.requestScan();

    // Try ESP32
    final success = await EspService.sendFaceSurgery(
      effectiveOption,
      widget.isDeaf,
    );

    setState(() {
      _isSending = false;
      _espStatus = success
          ? '✅ ESP32 received the command!'
          : '⚠️ ESP32 not connected — continuing anyway';
    });

    if (mounted) {
      await Future.delayed(const Duration(milliseconds: 600));
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => PreCheckScreen(
            category: 'Face Surgery',
            subCategory: effectiveOption,
            option: optionValue,
          ),
        ),
      );
    }
  }

  bool get _canConfirm {
    if (_selectedOption == null) return false;
    if (_selectedOption == 'mouth' && _selectedMouthOption == null) return false;
    return true;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.lightGrey,
      appBar: AppBar(
        backgroundColor: AppTheme.darkNavy,
        iconTheme: const IconThemeData(color: Colors.white),
        title: Row(
          children: [
            Text('Face Surgery',
                style: GoogleFonts.poppins(
                    color: Colors.white, fontWeight: FontWeight.w600)),
            if (widget.isDeaf) ...[
              const SizedBox(width: 8),
              _AppBarBadge(label: '🤟 Sign Language'),
            ],
            if (widget.beforeAfter) ...[
              const SizedBox(width: 6),
              _AppBarBadge(label: '🔄 Before/After'),
            ],
          ],
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 10),

            // ── Active Mode Banners ──────────────────────────────────
            if (widget.isDeaf)
              _ModeBanner(
                emoji: '🤟',
                text: 'Sign language mode is ON — the screen will activate sign language detection',
                color: AppTheme.accent,
              ),
            if (widget.beforeAfter) ...[
              if (widget.isDeaf) const SizedBox(height: 8),
              _ModeBanner(
                emoji: '🔄',
                text: 'Before/After mode is ON — the Jetson screen will show a comparison view',
                color: const Color(0xFF6A1B9A),
              ),
            ],

            const SizedBox(height: 16),
            Text('Select Area',
                style: GoogleFonts.poppins(
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.darkNavy)),
            Text('The filter will apply on the Jetson screen in real time',
                style: GoogleFonts.poppins(
                    fontSize: 13, color: AppTheme.textGrey)),
            const SizedBox(height: 20),

            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    ...List.generate(_options.length, (i) {
                      final opt = _options[i];
                      final isSelected = _selectedOption == opt['key'];
                      final isMouth = opt['key'] == 'mouth';

                      return Padding(
                        padding: const EdgeInsets.only(bottom: 16),
                        child: Column(
                          children: [
                            GestureDetector(
                              onTap: () => setState(() {
                                _selectedOption = opt['key'];
                                if (!isMouth) _selectedMouthOption = null;
                              }),
                              child: AnimatedContainer(
                                duration: const Duration(milliseconds: 200),
                                padding: const EdgeInsets.all(20),
                                decoration: BoxDecoration(
                                  color: isSelected
                                      ? AppTheme.blue
                                      : Colors.white,
                                  borderRadius: BorderRadius.circular(16),
                                  border: Border.all(
                                    color: isSelected
                                        ? AppTheme.blue
                                        : Colors.grey.shade200,
                                    width: 2,
                                  ),
                                  boxShadow: isSelected
                                      ? [
                                          BoxShadow(
                                            color: AppTheme.blue.withAlpha(76),
                                            blurRadius: 16,
                                            offset: const Offset(0, 4),
                                          )
                                        ]
                                      : [],
                                ),
                                child: Row(
                                  children: [
                                    Icon(opt['icon'] as IconData,
                                        color: isSelected
                                            ? Colors.white
                                            : AppTheme.blue,
                                        size: 30),
                                    const SizedBox(width: 16),
                                    Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(opt['label'] as String,
                                            style: GoogleFonts.poppins(
                                                fontSize: 16,
                                                fontWeight: FontWeight.w600,
                                                color: isSelected
                                                    ? Colors.white
                                                    : AppTheme.darkNavy)),
                                        Text(opt['desc'] as String,
                                            style: GoogleFonts.poppins(
                                                fontSize: 12,
                                                color: isSelected
                                                    ? Colors.white70
                                                    : AppTheme.textGrey)),
                                      ],
                                    ),
                                    const Spacer(),
                                    if (isMouth)
                                      Icon(
                                        isSelected
                                            ? Icons.expand_less
                                            : Icons.expand_more,
                                        color: isSelected
                                            ? Colors.white
                                            : AppTheme.blue,
                                      )
                                    else if (isSelected)
                                      const Icon(Icons.check_circle,
                                          color: Colors.white, size: 22),
                                  ],
                                ),
                              ),
                            ),

                            // ── Mouth Sub-options ──────────────────
                            if (isMouth && isSelected) ...[
                              const SizedBox(height: 8),
                              ...List.generate(_mouthOptions.length, (j) {
                                final mOpt = _mouthOptions[j];
                                final isMouthSelected =
                                    _selectedMouthOption == mOpt['key'];
                                return Padding(
                                  padding: const EdgeInsets.only(
                                      left: 16, bottom: 10),
                                  child: GestureDetector(
                                    onTap: () => setState(() =>
                                        _selectedMouthOption = mOpt['key']),
                                    child: AnimatedContainer(
                                      duration:
                                          const Duration(milliseconds: 200),
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 16, vertical: 14),
                                      decoration: BoxDecoration(
                                        color: isMouthSelected
                                            ? AppTheme.blue.withAlpha(25)
                                            : Colors.white,
                                        borderRadius:
                                            BorderRadius.circular(12),
                                        border: Border.all(
                                          color: isMouthSelected
                                              ? AppTheme.blue
                                              : Colors.grey.shade200,
                                          width: 1.5,
                                        ),
                                      ),
                                      child: Row(
                                        children: [
                                          Icon(mOpt['icon'] as IconData,
                                              color: isMouthSelected
                                                  ? AppTheme.blue
                                                  : AppTheme.textGrey,
                                              size: 24),
                                          const SizedBox(width: 12),
                                          Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(mOpt['label'] as String,
                                                  style: GoogleFonts.poppins(
                                                      fontSize: 14,
                                                      fontWeight:
                                                          FontWeight.w600,
                                                      color: isMouthSelected
                                                          ? AppTheme.blue
                                                          : AppTheme.darkNavy)),
                                              Text(mOpt['desc'] as String,
                                                  style: GoogleFonts.poppins(
                                                      fontSize: 11,
                                                      color: AppTheme.textGrey)),
                                            ],
                                          ),
                                          const Spacer(),
                                          if (isMouthSelected)
                                            Icon(Icons.check_circle,
                                                color: AppTheme.blue,
                                                size: 20),
                                        ],
                                      ),
                                    ),
                                  ),
                                );
                              }),
                            ],
                          ],
                        ),
                      );
                    }),

                    if (_espStatus != null) ...[
                      const SizedBox(height: 8),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: _espStatus!.contains('✅')
                              ? Colors.green.shade50
                              : Colors.orange.shade50,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                            color: _espStatus!.contains('✅')
                                ? Colors.green.shade200
                                : Colors.orange.shade200,
                          ),
                        ),
                        child: Text(_espStatus!,
                            style: GoogleFonts.poppins(fontSize: 13)),
                      ),
                    ],
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: (_canConfirm && !_isSending) ? _confirm : null,
                icon: _isSending
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                            color: Colors.white, strokeWidth: 2))
                    : const Icon(Icons.monitor_heart),
                label: Text(_isSending ? 'Sending...' : 'Pre-Check'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.blue,
                  disabledBackgroundColor: Colors.grey.shade300,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14)),
                ),
              ),
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }
}

// ── Helpers ───────────────────────────────────────────────────────

class _AppBarBadge extends StatelessWidget {
  final String label;
  const _AppBarBadge({required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: Colors.white.withAlpha(50),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(label,
          style: const TextStyle(color: Colors.white, fontSize: 11)),
    );
  }
}

class _ModeBanner extends StatelessWidget {
  final String emoji;
  final String text;
  final Color color;
  const _ModeBanner(
      {required this.emoji, required this.text, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withAlpha(25),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color, width: 1.5),
      ),
      child: Row(
        children: [
          Text(emoji, style: const TextStyle(fontSize: 20)),
          const SizedBox(width: 10),
          Expanded(
            child: Text(text,
                style: GoogleFonts.poppins(fontSize: 13, color: color)),
          ),
        ],
      ),
    );
  }
}