import tensorflow as tf

class Suspiciousness():
    def __init__(self, sus_values, sus_values_flat, shapes):
        self.sus_values = sus_values
        self.sus_values_flat = sus_values_flat
        self.shapes = shapes

    def get_sus_values(self):
        return self.sus_values

    def get_sus_values_flat(self):
        return self.sus_values_flat

    def get_shapes(self):
        return self.shapes

class HitSpectrum():
    def __init__(self, shape):
        self.shape = shape
        self.a_s = tf.Variable(tf.zeros(shape, dtype=tf.int32))
        self.a_f = tf.Variable(tf.zeros(shape, dtype=tf.int32))
        self.n_s = tf.Variable(tf.zeros(shape, dtype=tf.int32))
        self.n_f = tf.Variable(tf.zeros(shape, dtype=tf.int32))
    
    def get_values(self):
        return self.a_s, self.a_f, self.n_s, self.n_f
    
    # @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32), tf.TensorSpec(shape=[None], dtype=tf.int32)])
    # @tf.function
    def increment_as(self, results, intermediate_output):
        i = 0
        while i < len(results):
            self.a_s.assign_add(tf.math.multiply(results[i], intermediate_output[i]))
            i += 1

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
    # @tf.function
    def increment_af(self, results, intermediate_output):
        # and (not results, or results intermediate output)
        i = 0
        while i < len(results):
            self.a_f.assign_add(tf.where(tf.math.greater(intermediate_output[i], results[i]), 1, 0))
            i += 1
        return

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
    # @tf.function
    def increment_ns(self, results, intermediate_output):
        i = 0
        while i < len(results):
            self.n_s.assign_add(tf.where(tf.math.greater(results[i], intermediate_output[i]), 1, 0))
            i += 1
        return 

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
    # @tf.function
    def increment_nf(self, results, intermediate_output):
        # 
        i = 0
        while i < len(results):
            self.n_f.assign_add(tf.where(tf.math.greater(intermediate_output[i], results[i]), 1, 0))
            i += 1
        return 
    
    @tf.function
    def get_ochiai(self):
        numerator = tf.cast(self.a_f, dtype=tf.float32)
        denominator = tf.math.sqrt(tf.cast(tf.math.multiply(tf.math.add(self.a_f, self.n_f), tf.math.add(self.a_f, self.a_s)), dtype=tf.float32))
        
        return tf.math.divide_no_nan(numerator, denominator)
    
    @tf.function
    def get_tarantula(self):
        numerator = tf.cast(self.a_f, dtype=tf.float32) / tf.cast(self.a_f + self.n_f, dtype=tf.float32)
        denominator_l = tf.cast(self.a_f, dtype=tf.float32) / tf.cast(self.a_f + self.n_f, dtype=tf.float32)
        denominator_r = tf.cast(self.a_s, dtype=tf.float32) / tf.cast(self.a_s + self.n_s, dtype=tf.float32)
        return tf.math.divide_no_nan(numerator, denominator_l + denominator_r)
    
    @tf.function
    def get_d_star(self, star):
        numerator = tf.cast(self.a_f, dtype=tf.float32) ** star
        denominator = tf.cast(self.a_s + self.n_f, dtype=tf.float32)
        return tf.math.divide_no_nan(numerator, denominator)
