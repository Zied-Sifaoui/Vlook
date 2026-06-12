import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/firebase_service.dart';
import '../theme.dart';
import 'patient/patient_menu_screen.dart';
import 'doctor/doctor_dashboard_screen.dart';
import 'register_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;
  String? _error;

  Future<void> _login(String role) async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      if (role == 'doctor') {
        await FirebaseService.loginAsDoctor(
            _emailController.text.trim(), _passwordController.text.trim());
        if (mounted) {
          Navigator.pushReplacement(context,
              MaterialPageRoute(builder: (_) => const DoctorDashboardScreen()));
        }
      } else {
        await FirebaseService.loginAsPatient(
            _emailController.text.trim(), _passwordController.text.trim());
        if (mounted) {
          Navigator.pushReplacement(context,
              MaterialPageRoute(builder: (_) => const PatientMenuScreen()));
        }
      }
    } catch (e) {
      setState(() => _error = e.toString().replaceAll('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.lightGrey,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: 60),
              // ─── Logo ───────────────────────────────────────────
              Text(
                'MedAR',
                style: GoogleFonts.montserrat(
                  fontSize: 42,
                  fontWeight: FontWeight.w800,
                  color: AppTheme.darkNavy,
                  letterSpacing: 1.5,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Medical Augmented Reality',
                style: GoogleFonts.poppins(
                  fontSize: 14,
                  color: AppTheme.lightBlue,
                  fontWeight: FontWeight.w500,
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 60),
              // ─── Form Card ──────────────────────────────────────
              Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.06),
                      blurRadius: 20,
                      offset: const Offset(0, 4),
                    )
                  ],
                ),
                padding: const EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Email Address',
                        style: GoogleFonts.poppins(
                            fontWeight: FontWeight.w600,
                            color: AppTheme.darkNavy)),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _emailController,
                      keyboardType: TextInputType.emailAddress,
                      decoration: const InputDecoration(
                          hintText: 'doctor@hospital.com'),
                    ),
                    const SizedBox(height: 20),
                    Text('Password',
                        style: GoogleFonts.poppins(
                            fontWeight: FontWeight.w600,
                            color: AppTheme.darkNavy)),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _passwordController,
                      obscureText: true,
                      decoration:
                          const InputDecoration(hintText: '••••••••'),
                    ),
                    if (_error != null) ...[
                      const SizedBox(height: 12),
                      Text(_error!,
                          style: const TextStyle(color: Colors.red, fontSize: 13)),
                    ],
                    const SizedBox(height: 24),
                    if (_isLoading)
                      const Center(child: CircularProgressIndicator())
                    else ...[
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: () => _login('doctor'),
                          style: ElevatedButton.styleFrom(
                              backgroundColor: AppTheme.darkNavy),
                          child: const Text('Login as Doctor'),
                        ),
                      ),
                      const SizedBox(height: 12),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: () => _login('patient'),
                          style: ElevatedButton.styleFrom(
                              backgroundColor: AppTheme.lightBlue),
                          child: const Text('Login as Patient'),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
              const SizedBox(height: 24),
              // ─── Register Links ─────────────────────────────────
              TextButton(
                onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (_) =>
                            const RegisterScreen(role: 'patient'))),
                child: Text('New Patient? Register',
                    style: GoogleFonts.poppins(color: AppTheme.lightBlue)),
              ),
              TextButton(
                onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (_) =>
                            const RegisterScreen(role: 'doctor'))),
                child: Text('New Doctor? Register',
                    style: GoogleFonts.poppins(
                        color: AppTheme.darkNavy,
                        fontWeight: FontWeight.w600)),
              ),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }
}