import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme.dart';
import '../../services/firebase_service.dart';
import '../../services/esp_service.dart';
import 'pre_check_screen.dart';

class HairScreen extends StatefulWidget {
  final bool isDeaf;
  final bool beforeAfter;
  const HairScreen({
    super.key,
    required this.isDeaf,
    required this.beforeAfter,
  });

  @override
  State<HairScreen> createState() => _HairScreenState();
}

class _HairScreenState extends State<HairScreen> {
  String? _subCategory;
  String? _selectedOption;
  bool _isSending = false;
  String? _espStatus;

  final Map<String, List<Map<String, String>>> _options = {
    'hair': [
      {'key': 'style_1', 'label': 'Style 1 – Classic Cut'},
      {'key': 'style_2', 'label': 'Style 2 – Modern Fade'},
    ],
    'eyebrows': [
      {'key': 'brow_1', 'label': 'Natural Arch'},
      {'key': 'brow_2', 'label': 'Thick & Bold'},
      {'key': 'brow_3', 'label': 'Thin & Defined'},
      {'key': 'brow_4', 'label': 'Straight Brow'},
    ],
  };

  Future<void> _confirm() async {
    if (_selectedOption == null || _subCategory == null) return;
    setState(() {
      _isSending = true;
      _espStatus = null;
    });

    // Build option string — includes beforeAfter flag if enabled
    final optionValue = widget.isDeaf
        ? 'sign_language'
        : widget.beforeAfter
            ? '${_selectedOption!}_before_after'
            : _selectedOption!;

    // Send to Firebase / Jetson
    await FirebaseService.sendFilterToJetson(
      category: 'hair',
      subCategory: _subCategory!,
      option: optionValue,
      beforeAfter: widget.beforeAfter,
    );

    // Trigger the Python scan
    await FirebaseService.requestScan();

    // Try ESP32
    final success = await EspService.sendHair(
      _subCategory!,
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
            category: 'Hair Implementation',
            subCategory: _subCategory!,
            option: optionValue,
          ),
        ),
      );
    }
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
            Text('Hair Implementation',
                style: GoogleFonts.poppins(
                    color: Colors.white, fontWeight: FontWeight.w600)),
            if (widget.isDeaf) ...[
              const SizedBox(width: 8),
              _AppBarBadge(label: '🤟'),
            ],
            if (widget.beforeAfter) ...[
              const SizedBox(width: 6),
              _AppBarBadge(label: '🔄'),
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
            Text('Choose Type',
                style: GoogleFonts.poppins(
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.darkNavy)),
            const SizedBox(height: 20),

            Row(
              children: [
                _CategoryChip(
                  label: 'Hair',
                  icon: Icons.cut,
                  selected: _subCategory == 'hair',
                  onTap: () => setState(() {
                    _subCategory = 'hair';
                    _selectedOption = null;
                  }),
                ),
                const SizedBox(width: 12),
                _CategoryChip(
                  label: 'Eyebrows',
                  icon: Icons.remove,
                  selected: _subCategory == 'eyebrows',
                  onTap: () => setState(() {
                    _subCategory = 'eyebrows';
                    _selectedOption = null;
                  }),
                ),
              ],
            ),
            const SizedBox(height: 24),

            if (_subCategory != null) ...[
              Text(
                _subCategory == 'hair'
                    ? 'Choose Style'
                    : 'Choose Eyebrow Style',
                style: GoogleFonts.poppins(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: AppTheme.darkNavy),
              ),
              const SizedBox(height: 12),
              Expanded(
                child: ListView(
                  children: _options[_subCategory!]!.map((opt) {
                    final isSelected = _selectedOption == opt['key'];
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: GestureDetector(
                        onTap: () =>
                            setState(() => _selectedOption = opt['key']),
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 200),
                          padding: const EdgeInsets.symmetric(
                              horizontal: 20, vertical: 18),
                          decoration: BoxDecoration(
                            color: isSelected ? AppTheme.accent : Colors.white,
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(
                              color: isSelected
                                  ? AppTheme.accent
                                  : Colors.grey.shade200,
                              width: 2,
                            ),
                            boxShadow: isSelected
                                ? [
                                    BoxShadow(
                                      color: AppTheme.accent.withAlpha(76),
                                      blurRadius: 12,
                                      offset: const Offset(0, 4),
                                    )
                                  ]
                                : [],
                          ),
                          child: Row(
                            children: [
                              Text(opt['label']!,
                                  style: GoogleFonts.poppins(
                                      fontSize: 15,
                                      fontWeight: FontWeight.w500,
                                      color: isSelected
                                          ? Colors.white
                                          : AppTheme.darkNavy)),
                              const Spacer(),
                              if (isSelected)
                                const Icon(Icons.check_circle,
                                    color: Colors.white, size: 20),
                            ],
                          ),
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ),
            ] else
              const Expanded(child: SizedBox()),

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

            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed:
                    (_selectedOption != null && !_isSending) ? _confirm : null,
                icon: _isSending
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                            color: Colors.white, strokeWidth: 2))
                    : const Icon(Icons.monitor_heart),
                label: Text(_isSending ? 'Sending...' : 'Pre-Check'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.accent,
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
          style: const TextStyle(color: Colors.white, fontSize: 13)),
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

class _CategoryChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool selected;
  final VoidCallback onTap;

  const _CategoryChip({
    required this.label,
    required this.icon,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        decoration: BoxDecoration(
          color: selected ? AppTheme.darkNavy : Colors.white,
          borderRadius: BorderRadius.circular(30),
          border: Border.all(
            color: selected ? AppTheme.darkNavy : Colors.grey.shade300,
          ),
        ),
        child: Row(
          children: [
            Icon(icon,
                size: 18,
                color: selected ? Colors.white : AppTheme.textGrey),
            const SizedBox(width: 8),
            Text(label,
                style: GoogleFonts.poppins(
                    fontWeight: FontWeight.w600,
                    color: selected ? Colors.white : AppTheme.darkNavy)),
          ],
        ),
      ),
    );
  }
}