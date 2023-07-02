import 'package:div/pages/login.dart';
import 'package:flutter/material.dart';

class regform extends StatefulWidget {
  @override
  State<regform> createState() => _regformState();
}

class _regformState extends State<regform> {
  final _formKey = GlobalKey<FormState>();

  String? _name;
  String? _contact;
  String? _email;
  String? _address;
  String? _id;
  String? _profession;
  String? _locality;
  String? _availability;

  void _submitForm() {
    if (_formKey.currentState!.validate()) {
      //TODO: submit form
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        backgroundColor: Colors.green[50],
        body: SingleChildScrollView(
          child: Form(
            key: _formKey,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 30),
              child: Column(
                children: [
                  SizedBox(height: 100),
                  Container(
                    padding: EdgeInsets.all(10),
                    decoration: BoxDecoration(
                        color: Colors.green,
                        borderRadius: BorderRadius.circular(15)),
                    child: Text(
                      "Registration Form",
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  TextFormField(
                    decoration: InputDecoration(labelText: 'Name'),
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your name';
                      }
                      return null;
                    },
                    onSaved: (value) {
                      _name = value;
                    },
                  ),
                  TextFormField(
                    decoration: InputDecoration(labelText: 'Contact'),
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your contact number';
                      }
                      return null;
                    },
                    onSaved: (value) {
                      _contact = value;
                    },
                  ),
                  TextFormField(
                    decoration: InputDecoration(labelText: 'Email'),
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your email address';
                      }
                      return null;
                    },
                    onSaved: (value) {
                      _email = value;
                    },
                  ),
                  TextFormField(
                    decoration: InputDecoration(labelText: 'Address'),
                    onSaved: (value) {
                      _address = value;
                    },
                  ),
                  TextFormField(
                    decoration: InputDecoration(labelText: 'ID'),
                    onSaved: (value) {
                      _id = value;
                    },
                  ),
                  TextFormField(
                    decoration: InputDecoration(labelText: 'Profession'),
                    onSaved: (value) {
                      _profession = value;
                    },
                  ),
                  TextFormField(
                    decoration: InputDecoration(labelText: 'Locality'),
                    onSaved: (value) {
                      _locality = value;
                    },
                  ),
                  TextFormField(
                    decoration: InputDecoration(labelText: 'Availability'),
                    onSaved: (value) {
                      _availability = value;
                    },
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  ElevatedButton(
                    style: ButtonStyle(
                        backgroundColor:
                            MaterialStateProperty.all(Colors.green)),
                    onPressed: () {
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => Login(),
                          ));
                    },
                    child: Text('Submit'),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
