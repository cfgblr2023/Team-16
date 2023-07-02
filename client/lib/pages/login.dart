import 'package:div/pages/regis.dart';
import 'package:div/widgets/button.dart';
import 'package:div/widgets/login_text.dart';
import 'package:flutter/material.dart';

class Login extends StatelessWidget {
  //text
  final username = TextEditingController();
  final password = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.green[50],
      body: SingleChildScrollView(
        child: SafeArea(
          child: Center(
            child: Column(
              children: [
                SizedBox(height: 100),

                //icon on top
                Icon(
                  Icons.person_2,
                  size: 100,
                ),
                SizedBox(
                  height: 30,
                ),
                Text(
                  "Login Here",
                  style: TextStyle(fontSize: 20),
                ),
                SizedBox(
                  height: 30,
                ),

                //email field
                login_text(
                  controller: username,
                  text_hint: 'Enter Username',
                  text_obs: false,
                ),
                SizedBox(
                  height: 20,
                ),

                //password field
                login_text(
                  controller: password,
                  text_hint: 'Enter Password',
                  text_obs: true,
                ),

                SizedBox(
                  height: 10,
                ),

                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 26),
                  child: Row(
                    children: [
                      Text(
                        "Forgot Password?",
                        style: TextStyle(color: Colors.black),
                      ),
                    ],
                  ),
                ),

                SizedBox(
                  height: 20,
                ),

                //sign in button
                button(),

                SizedBox(
                  height: 10,
                ),

                //not a member
                InkWell(
                  onTap: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => regform(),
                        ));
                  },
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        "New User?",
                        style: TextStyle(),
                      ),
                      SizedBox(width: 5),
                      Text(
                        "Sign Up",
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
