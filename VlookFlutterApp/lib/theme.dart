import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  static const Color darkNavy = Color(0xFF0D1B2A);
  static const Color navy = Color(0xFF1B2A3F);
  static const Color blue = Color(0xFF1565C0);
  static const Color lightBlue = Color(0xFF1E88E5);
  static const Color accent = Color(0xFF00BCD4);
  static const Color white = Color(0xFFFFFFFF);
  static const Color lightGrey = Color(0xFFF0F4F8);
  static const Color textGrey = Color(0xFF607D8B);

  static ThemeData get theme => ThemeData(
        scaffoldBackgroundColor: lightGrey,
        colorScheme: ColorScheme.fromSeed(
          seedColor: blue,
          primary: blue,
          secondary: accent,
        ),
        textTheme: GoogleFonts.poppinsTextTheme(),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: blue,
            foregroundColor: white,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            padding: const EdgeInsets.symmetric(vertical: 16),
            textStyle: GoogleFonts.poppins(
                fontSize: 16, fontWeight: FontWeight.w600),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: blue, width: 2),
          ),
        ),
      );
}