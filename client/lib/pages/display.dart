import 'package:div/pages/login.dart';
import 'package:flutter/material.dart';

class displayArea extends StatelessWidget {
  const displayArea({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.green[50],
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'YOUR DETAILS HAVE BEEN RECORDED',
              style: TextStyle(
                color: Colors.green,
                fontSize: 20,
              ),
            ),
            SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton(
                  child: Text('Okay'),
                  style: ButtonStyle(
                    backgroundColor: MaterialStateProperty.all(Colors.green),
                  ),
                  onPressed: () {
                    // Do something when the button is pressed.
                  },
                ),
                ElevatedButton(
                  child: Text('Logout'),
                  style: ButtonStyle(
                    backgroundColor: MaterialStateProperty.all(Colors.red),
                  ),
                  onPressed: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => Login(),
                        ));
                    // Do something when the button is pressed.
                  },
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
