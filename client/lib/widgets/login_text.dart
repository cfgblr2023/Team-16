import 'package:flutter/material.dart';

class login_text extends StatelessWidget {
  final controller;
  final String text_hint;
  final bool text_obs;

  login_text({this.controller, required this.text_hint, required this.text_obs});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 25.0),
      child: TextField(
        controller: controller,
        decoration: InputDecoration(
            enabledBorder: const OutlineInputBorder(
              borderSide: BorderSide(color: Colors.green),
            ),
            focusedBorder: OutlineInputBorder(
              borderSide: BorderSide(color: Colors.green),
            ),
            fillColor: Colors.grey.shade200,
            filled: true,
            hintText: text_hint,
            hintStyle: TextStyle(color: Colors.grey)),
            obscureText: text_obs,
      ),
    );
  }
}
