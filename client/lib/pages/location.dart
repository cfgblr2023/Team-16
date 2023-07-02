import 'package:div/pages/display.dart';
import 'package:div/pages/regis.dart';
import 'package:flutter/material.dart';
import 'package:open_street_map_search_and_pick/open_street_map_search_and_pick.dart';

class MapLocation extends StatefulWidget {
  const MapLocation({super.key});

  @override
  State<MapLocation> createState() => _MapLocationState();
}

class _MapLocationState extends State<MapLocation> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.green[50],
      body: OpenStreetMapSearchAndPick(
          locationPinIconColor: Colors.green,
          center: LatLong(13.059021099999999, 80.28138336226445),
          buttonColor: Colors.blue,
          buttonText: 'Set Current Location',
          onPicked: (pickedData) {
            print(pickedData.latLong.latitude);
            print(pickedData.latLong.longitude);
            print(pickedData.address);
            Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => displayArea(),
            ));

          }),
    );
  }
}
