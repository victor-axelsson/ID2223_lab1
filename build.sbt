name := "lab1"

organization := "se.kth.spark"

version := "1.0"

scalaVersion := "2.11.1"

////resolvers += Resolver.mavenLocal
resolvers += "Kompics Snapshots" at "http://kompics.sics.se/maven/snapshotrepository/"
resolvers +=
  "imagej" at "http://maven.imagej.net/content/repositories/releases/"


libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.1"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.1"
libraryDependencies += "org.log4s" %% "log4s" % "1.3.3" % "provided"
libraryDependencies += "se.kth.spark" %% "lab1_lib" % "1.0-SNAPSHOT"
libraryDependencies += "cisd" % "jhdf5" % "12.02.3"