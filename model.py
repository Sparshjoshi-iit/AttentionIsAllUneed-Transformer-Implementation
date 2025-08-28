import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) containing token indices.
        
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with embedded tokens.
        """
        x = self.embedding(x)
        return x * (self.d_model ** 0.5)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create the positional encodings matrix
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  #createing a column vector of shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe= pe.unsqueeze(0)  # Add batch dimension
        
        # Register the positional encodings as a buffer so they are not considered model parameters    
        self.register_buffer('pe', pe.unsqueeze(0))        
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self,eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.layer_norm = nn.Parameter(torch.ones(1))   #using a learnable parameter for layer normalization by writing .Parameter
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor of the same shape with layer normalization applied.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)       
        normalized_x = (x - mean) / (std + self.eps)
        return self.layer_norm * normalized_x + self.bias
    
class FeedForward(nn.Module):
        def __init__(self, d_model: int, d_ff: int, dropout: float):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff) #Weights for the first linear transformation
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ff, d_model) #Weights for the second linear transformation
            
        def forward(self, x):
            """
            Args:
                x: Tensor of shape (batch_size, seq_len, d_model).
            Process:
                (Batch size, Sequence length, Model dimension) -> (Batch size, Sequence length, Feed-forward dimension) -> (Batch size, Sequence length, Model dimension)
            Returns:
                Tensor of shape (batch_size, seq_len, d_model) after applying the feed-forward network.
            """
            return self.linear2(self.dropout(torch.relu(self.linear1(x))))
         
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.d_k= self.d_model // self.n_heads
        self.w_q=nn.Linear(d_model, d_model)  #Weights for the query transformation
        self.w_k=nn.Linear(d_model, d_model)
        self.w_v=nn.Linear(d_model, d_model)
        self.w_o=nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)  
    
    @staticmethod
    def attention(query, key, value,mask,dropout:nn.Dropout):
        d_k= query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / d_k ** 0.5
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, value), attention_scores
    
    def forward(self, query, key, value, mask=None):
        query= self.w_q(query)
        key= self.w_k(key)
        value= self.w_v(value)
        batch_size, seq_len, _ = query.size()
        
        query=query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        key=key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value=value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        x,self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        x=x.transpose(1, 2).contiguous().view(x.shape[0],-1,self.n_heads * self.d_k)  # (batch_size, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm= LayerNormalization()
        
    def forward(self, x, sublayer_output):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).
            sublayer_output: Tensor of the same shape as x.
        
        Returns:
            Tensor of the same shape as x after applying the residual connection and dropout.
        """
        return self.dropout(x + sublayer_output)  # Adding the original input to the output of the sublayer

class EncoderBlock(nn.Module):
    
    def __init__(self,self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))  # Two residual connections: one for attention and one for feed-forward
    
    def forward(self, x, mask):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).
            mask: Optional mask tensor for attention.
        
        Returns:
            Tensor of the same shape as x after applying self-attention and feed-forward network.
        """
        x=self.residual_connections[0](x,lambda x: self.self_attention(x, x, x, mask))  # Self-attention
        x=self.residual_connections[1](x, self.feed_forward(x))  # Feed-forward network
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm=LayerNormalization()  # Final layer normalization
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init(self,self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))
        
    def forward(self,x,encoder_output,mask,target_mask):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).
            encoder_output: Output from the encoder of shape (batch_size, seq_len, d_model).
            mask: Optional mask tensor for self-attention.
            target_mask: Optional mask tensor for cross-attention.
        
        Returns:
            Tensor of the same shape as x after applying self-attention, cross-attention, and feed-forward network.
        """
        x=self.residual_connections[0](x,lambda x: self.self_attention(x, x, x, target_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_attention(x, encoder_output, encoder_output, mask))
        x=self.residual_connections[2](x, self.feed_forward(x))
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()  # Final layer normalization
        
    def forward(self, x, encoder_output, mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, mask, target_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self,x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size) after applying the linear transformation.
        """
        return torch.log_softmax(self.linear(x),dim=-1)
    
class TransformerBlock(nn.Module):
    def __init__(self,encoder: Encoder, decoder: Decoder, input_embedding: InputEmbedding, positional_encoding: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.positional_encoding = positional_encoding
        self.projection_layer = projection_layer
    def encode(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source input tensor of shape (batch_size, src_seq_len).
            tgt: Target input tensor of shape (batch_size, tgt_seq_len).
            src_mask: Optional mask for source input.
            tgt_mask: Optional mask for target input.
        Returns:
            Tensor of shape (batch_size, tgt_seq_len, vocab_size) after passing through the transformer block.
        """
        src=self.input_embedding(src)
        src=self.positional_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            encoder_output: Output from the encoder of shape (batch_size, src_seq_len, d_model).
            tgt: Target input tensor of shape (batch_size, tgt_seq_len).
            src_mask: Optional mask for source input.
            tgt_mask: Optional mask for target input.
        Returns:
            Tensor of shape (batch_size, tgt_seq_len, vocab_size) after passing through the transformer block.
        """
        tgt=self.input_embedding(tgt)
        tgt=self.positional_encoding(tgt)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.projection_layer(decoder_output)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> TransformerBlock:
    # Create the embedding layer
    src= InputEmbedding(d_model, src_vocab_size)
    tgt = InputEmbedding(d_model, tgt_vocab_size)
    # Create the positional encoding layer
    src_pe = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pe = PositionalEncoding(d_model, tgt_seq_len, dropout)
    # Create the multi-head attention layers
    encoder_blocks=[]
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(self_attention, feed_forward, dropout))
    
    decoder_blocks=[]
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        cross_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(self_attention, cross_attention, feed_forward, dropout))
    
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transfomer= TransformerBlock(encoder, decoder, src, src_pe, projection_layer)
    for p in transfomer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transfomer
