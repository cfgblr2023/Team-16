import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as path;

// Define the Flask API endpoint URL
final String flaskApiUrl =
    'https://4a6d-165-225-106-79.in.ngrok.io'; // Replace with your Flask API URL

// Define the path to the image file
final String imagePath =
    './div/lib/photo.png'; // Replace with the actual path to your image file

// Make a POST request to the Flask API
Future<void> callFlaskAPI() async {
  Uri uri = Uri.parse(flaskApiUrl);

  // Create a multipart request
  var request = http.MultipartRequest('POST', uri);

  // Add the image file to the request
  request.files.add(await http.MultipartFile.fromPath(
    'image',
    imagePath,
    contentType: MediaType('image', path.extension(imagePath)),
  ));

  try {
    var response = await request.send();

    // Check if the request was successful
    if (response.statusCode == 200) {
      var responseBody = await response.stream.bytesToString();
      // Handle the response
      print('Predicted Class: $responseBody');
    } else {
      // Handle the error
      print('Error: ${response.statusCode}');
    }
  } catch (error) {
    // Handle the error
    print('Error: $error');
  }
}
