import java.io.IOException;
import java.util.*;
import java.lang.NumberFormatException;
import java.lang.Math;
import java.io.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class KMean {


    public static class MapperSetup extends Mapper<LongWritable, Text, IntWritable, StringToTextWritable> { //initiation mapper for first iteration

        private int clusters; //number of clusters
        private final Random random = new Random(); //random generator

        protected void setup(Context context) throws IOException, InterruptedException { //setup for first iteration
            Configuration config = context.getConfiguration();
            clusters =  Integer.parseInt(config.get("Clusters")); //get the number of clusters from the context
        }
        public void map(LongWritable id, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString(); //each line in the input file
            String[] tokens = line.split(","); //tokens in each line where first 4 tokens are the 4 dimensions of the point and 5th is the classification

            if(tokens!= null && tokens.length == 5) //validation
                context.write(new IntWritable(1 + random.nextInt(clusters)) , new StringToTextWritable(tokens)); //assign random cluster id to each point either 1, 2 or 3
        }
    }
    public static class ReducerSetup extends Reducer<IntWritable, StringToTextWritable, Text, Text> { //initiation reducer for first iteration

        private double[][] centroid ;
        private String output = "";
        private String[] NamesOfClusters ; //for clusters included in the centroid
        private String lastPoint = "";
        private int clusters; //number of clusters
        private int iterate = 0; //current iteration


        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration config = context.getConfiguration();
            output = config.get("Output"); //output file for saving
            clusters =  Integer.parseInt(config.get("Clusters"));
            centroid = new double[clusters][4]; //number of centroid is same number of clusters
            NamesOfClusters = new String[clusters];
            for(int i = 0; i < NamesOfClusters.length; i++){
                NamesOfClusters[i] = ""; //initiation
            }
        }
        public void reduce(IntWritable id, Iterable<StringToTextWritable> values, Context context) throws IOException, InterruptedException {
            int totalPoints = 0; //counter
            double[] center = new double[4]; //center of current iteration
            for(int i = 0; i < center.length; i++){
                center[i] = 0; //initiation
            }
            int key = id.get() - 1; //clusters IDs are 1,2,3 but array indices are 0,1,2
            for (StringToTextWritable value : values) {
                String[] point = value.toStrings(); //current point
                if(!NamesOfClusters[key].contains(point[4])){ //get the cluster name from the classification
                    NamesOfClusters[key] = NamesOfClusters[key] + point[4] + "+";
                }
                totalPoints ++;

                String data = ""; //data of each point
                for(int i = 0; i < point.length - 1; i++){ //the 4 dimensions of each point without the classification
                    data = data + point[i] + ","; //all 4 dimension
                    center[i] += Double.parseDouble(point[i]); //center summation for new centroid
                }
                lastPoint = data;
                context.write(new Text(id.toString() + ",") , new Text(data + point[4])); //cluster id followed by the point and classification
            }
            for(int i = 0; i < center.length; i++){
                centroid[key][i] = center[i] / totalPoints; //new centroid
            }

        }
        protected void cleanup(Context context) throws IOException,InterruptedException{
            Configuration config = context.getConfiguration();
            config.set("lastPoint" , lastPoint );
            updateCentroid(centroid , iterate , output , NamesOfClusters); //update new centroid
        }
    }

    public static class TokenizerMapper extends Mapper<LongWritable, Text, IntWritable, StringToTextWritable> { //mapper for rest of iterations till convergence

        private int iterate = 0;
        private int clusters = 0;
        private double[][] centroid;
        private String output = "";

        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration config = context.getConfiguration();
            iterate = Integer.parseInt(config.get("Iterate"));
            clusters =  Integer.parseInt(config.get("Clusters"));
            output = config.get("Output");
            centroid = getCentroid(clusters , iterate , output); //get centroid of last iteration
        }

        public void map(LongWritable id, Text value, Context context) throws IOException, InterruptedException {
            double[] point = new double[4];
            String line = value.toString();
            String[] tokens = line.split(",");
            int i = 0;
            tokens[1] = tokens[1].replaceAll("\\s+"," "); //for the whitespace between id and point
            for(i = 0; i < 4; i++){
                point[i] = Double.parseDouble(tokens[i+1]); //read point from string
            }
            int clusterId = 0;
            double min = 0;
            double sum;
            for(i = 0; i < centroid.length; i++){ //for each old centroid
                sum = 0;
                for(int j = 0; j < 4; j++){
                    sum += ((centroid[i][j] - point[j]) * (centroid[i][j] - point[j])); //calculate euclidean distance from point to centroid
                }
                if(min == 0){ //first point
                    min = sum;
                    clusterId = i + 1;
                }
                else if(sum < min){
                    min = sum; //new centroid is assigned
                    clusterId = i + 1; //new cluster id
                }
            }
            context.write(new IntWritable(clusterId) , new StringToTextWritable(Arrays.copyOfRange(tokens , 1 , tokens.length))); //cluster id with the point without old cluster id
        }
    }

    public static class IntSumReducer extends Reducer<IntWritable, StringToTextWritable, Text, Text> {//reducer for rest of iterations till convergence

        private int iterate = 0;
        private double[][] centroid ;
        private int clusters;
        private String[] NamesOfClusters;
        private String output = "";
        private String lastPoint;

        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration config = context.getConfiguration();

            iterate = Integer.parseInt(config.get("Iterate"));
            clusters =  Integer.parseInt(config.get("Clusters"));
            centroid = new double[clusters][4];
            output = config.get("Output");
            NamesOfClusters = new String[clusters];
            for(int i = 0; i < NamesOfClusters.length; i++){
                NamesOfClusters[i] = ""; //initiation
            }
            lastPoint = config.get("lastPoint");
        }
        public void reduce(IntWritable id, Iterable<StringToTextWritable> values, Context context) throws IOException, InterruptedException {
            double[] center = new double[4];
            int key = id.get() - 1;
            Iterator<StringToTextWritable> iterator = values.iterator();
            if(!iterator.hasNext()){
                String[] d = lastPoint.split(",");
                for(int i = 0; i < d.length; i++){
                    centroid[key][i] = Double.parseDouble(d[i]);
                }
                context.write(new Text(id.toString() + ",") , new Text(lastPoint));
            }
            else{
                for(int i = 0; i < center.length; i++){
                    center[i] = 0; //initiation
                }
                int valSize = 0;
                for (StringToTextWritable val : values) {
                    String[] point = val.toStrings();
                    if(!NamesOfClusters[key].contains(point[4])){
                        NamesOfClusters[key] = NamesOfClusters[key] + point[4] + "+"; //new classification included in this cluster
                    }
                    valSize ++;
                    String data = "";
                    for(int i = 0; i < point.length - 1; i++){
                        data = data +  point[i] + ","; //concatenate points to string
                        center[i] += Double.parseDouble(point[i]); //new centroid
                    }
                    lastPoint = data + point[4];
                    context.write(new Text(id.toString() + ",") , new Text(data + point[4]));
                }
                for(int i = 0; i < center.length; i++){
                    centroid[key][i] = center[i] / valSize;
                }
            }
        }
        protected void cleanup(Context context) throws IOException,InterruptedException{
            Configuration config = context.getConfiguration();
            config.set("lastPoint" , lastPoint );
            updateCentroid(centroid , iterate , output , NamesOfClusters);
        }
    }

    public static class StringToTextWritable extends ArrayWritable{ //convert array of strings to array of texts

        public StringToTextWritable(){
            super(Text.class);
        }

        public StringToTextWritable(String[] points) {
            super(Text.class);
            Text[] texts = new Text[points.length];
            for (int i = 0; i < points.length; i++) {
                texts[i] = new Text(points[i]);
            }
            set(texts);
        }
    }

    public static void main(String[] args) throws Exception {
        String clusters = "3"; //number of clusters
        String input = "";
        String output = "";
        int iterate = 0;
        if(args.length == 2){
            input = args[0];
            output = args[1];
        }
        else{
            System.exit(0);
        }
        //for first iteration assign random cluster id and compute centroid
        long startTime = System.nanoTime();
        Configuration conf_init = new Configuration();
        conf_init.set("Clusters" , clusters );
        conf_init.set("Output" , output);
        Job job_init = new Job(conf_init, "KMeanSetup");
        job_init.setMapOutputKeyClass(IntWritable.class);
        job_init.setMapOutputValueClass(StringToTextWritable.class);

        job_init.setOutputKeyClass(Text.class);
        job_init.setOutputValueClass(Text.class);

        job_init.setMapperClass(MapperSetup.class);//initial mapper
        job_init.setReducerClass(ReducerSetup.class);//initial reducer
        job_init.setJarByClass(KMean.class);

        job_init.setInputFormatClass(TextInputFormat.class);
        job_init.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job_init, new Path(input));
        FileOutputFormat.setOutputPath(job_init, new Path(output + "/iterate" + Integer.toString(iterate)));
        job_init.waitForCompletion(true);
        String lastPoint = conf_init.get("lastPoint");
        Configuration conf = new Configuration();
        //for rest of iteration til convergence
        while(!converge(Integer.parseInt(clusters), iterate, output)){
            iterate++;
            conf.set("Iterate" , Integer.toString(iterate));
            conf.set("Clusters" , clusters);
            conf.set("Output" , output);
            Job job = new Job(conf, "KMeans");
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(StringToTextWritable.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            job.setMapperClass(TokenizerMapper.class);
            job.setReducerClass(IntSumReducer.class);
            job.setJarByClass(KMean.class);

            job.setInputFormatClass(TextInputFormat.class);
            job.setOutputFormatClass(TextOutputFormat.class);

            FileInputFormat.addInputPath(job, new Path(output + "/iterate" + Integer.toString(iterate - 1) + "/part-r-00000"));
            FileOutputFormat.setOutputPath(job, new Path(output + "/iterate" + Integer.toString(iterate)));
            job.waitForCompletion(true);
        }

        long endTime = System.nanoTime();
        System.out.println("Time taken: " + ((endTime - startTime) / 1000000) + " ms");

    }

    public static boolean converge(int clusters , int index ,String dir) throws IOException { //check for convergence
        if(index == 0) { //first iteration
            return false;
        }
        else if  (index == 50){ //maximum number of iterations achieved
            return true;
        }
        double[][] centroid = getCentroid(clusters, index++, dir); //current centroid
        double[][] centroidOld = getCentroid(clusters, index, dir);//previous centroid

        for(int i = 0; i < clusters; i++) {
            double euclidean;
            for(int j = 0; j < centroid[i].length;j++) {
                euclidean = (centroid[i][j] - centroidOld[i][j]) * (centroid[i][j] - centroidOld[i][j]);
                if(euclidean > 0.01) //meets the threshold
                    return false;
            }
        }
        return true;
    }

    public static void updateCentroid(double[][] points, int index , String dir , String[] NamesOfClusters){
        try{
            Configuration config = new Configuration();
            FileSystem hdfs = FileSystem.get(config);
            Path centroidFile = new Path(dir + "/centroid" + Integer.toString(index)); //path of new centroid file

            OutputStream out;
            out = hdfs.create(centroidFile);
            OutputStreamWriter osw = new OutputStreamWriter(out);
            for(int i = 0 ;i < points.length; i++){
                for(int j = 0; j < points[i].length; j++){
                    osw.append(Double.toString(points[i][j]) + ","); //write the point
                }
                if (NamesOfClusters[i].length() > 0){
                    NamesOfClusters[i] = NamesOfClusters[i].substring(0, NamesOfClusters[i].length() - 1); //write clusters included in the centroid
                }
                osw.append(NamesOfClusters[i]);
                osw.append("\n");
            }
            System.out.print("\n");
            osw.flush();
            osw.close();
        }catch(Exception e){
            System.out.println("error updating the centroid");
        }

    }

    public static double[][] getCentroid(int clusters , int index , String dir){
        double[][] centroid = new double[clusters][4];
        try{
            Configuration config = new Configuration();
            FileSystem hdfs = FileSystem.get(config);
            InputStreamReader isr = new InputStreamReader(hdfs.open(new Path(dir + "/centroid" + Integer.toString(index - 1)))); //read centroid of previous iteration
            BufferedReader br = new BufferedReader(isr);
            String line = br.readLine();
            int pos = 0;
            while(line != null){
                String[] tokens = line.split(",");
                for(int i = 0; i < tokens.length - 1; i++){
                    centroid[pos][i] = Double.parseDouble(tokens[i]); //convert string centroid to double
                }
                line = br.readLine();
                pos++;
            }
            br.close();
        }catch(Exception e){
            System.out.println("error opening the file");
            return null;
        }
        return centroid;
    }
}
