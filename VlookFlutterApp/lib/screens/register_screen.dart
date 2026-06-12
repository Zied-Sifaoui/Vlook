import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/firebase_service.dart';
import '../theme.dart';

class RegisterScreen extends StatefulWidget {
  final String role; // 'doctor' or 'patient'
  const RegisterScreen({super.key, required this.role});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;
  String? _error;

  Future<void> _register() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      if (widget.role == 'doctor') {
        await FirebaseService.registerDoctor(
          _emailController.text.trim(),
          _passwordController.text.trim(),
          _nameController.text.trim(),
        );
      } else {
        await FirebaseService.registerPatient(
          _emailController.text.trim(),
          _passwordController.text.trim(),
          _nameController.text.trim(),
        );
      }
      if (mounted) Navigator.pop(context);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDoctor = widget.role == 'doctor';
    return Scaffold(
      backgroundColor: AppTheme.lightGrey,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: AppTheme.darkNavy),
        title: Text(
          isDoctor ? 'Doctor Registration' : 'Patient Registration',
          style: GoogleFonts.poppins(
              color: AppTheme.darkNavy, fontWeight: FontWeight.w600),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            const SizedBox(height: 20),
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
                  _label('Full Name'),
                  const SizedBox(height: 8),
                  TextField(
                      controller: _nameController,
                      decoration:
                          const InputDecoration(hintText: 'John Doe')),
                  const SizedBox(height: 20),
                  _label('Email Address'),
                  const SizedBox(height: 8),
                  TextField(
                      controller: _emailController,
                      keyboardType: TextInputType.emailAddress,
                      decoration: const InputDecoration(
                          hintText: 'example@hospital.com')),
                  const SizedBox(height: 20),
                  _label('Password'),
                  const SizedBox(height: 8),
                  TextField(
                      controller: _passwordController,
                      obscureText: true,
                      decoration:
                          const InputDecoration(hintText: '••••••••')),
                  if (_error != null) ...[
                    const SizedBox(height: 12),
                    Text(_error!,
                        style:
                            const TextStyle(color: Colors.red, fontSize: 13)),
                  ],
                  const SizedBox(height: 24),
                  if (_isLoading)
                    const Center(child: CircularProgressIndicator())
                  else
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: _register,
                        style: ElevatedButton.styleFrom(
                          backgroundColor:
                              isDoctor ? AppTheme.darkNavy : AppTheme.lightBlue,
                        ),
                        child: Text(
                            'Register as ${isDoctor ? 'Doctor' : 'Patient'}'),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _label(String text) => Text(
        text,
        style: GoogleFonts.poppins(
            fontWeight: FontWeight.w600, color: AppTheme.darkNavy),
      );
}