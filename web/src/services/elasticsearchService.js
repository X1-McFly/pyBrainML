import { Client } from '@elasticsearch/elasticsearch';

export class ElasticsearchService {
  constructor(config) {
    this.config = config;
    this.client = new Client({
      node: config.node,
      auth: {
        username: config.username,
        password: config.password
      },
      // Disable SSL verification for local development
      tls: {
        rejectUnauthorized: false
      }
    });
    this.index = config.index || 'eeg_data';
  }

  async testConnection() {
    try {
      const response = await this.client.ping();
      return response;
    } catch (error) {
      throw new Error(`Connection failed: ${error.message}`);
    }
  }

  async fetchIncrementalData(experimentId, lastSequenceId = 0, size = 1000) {
    try {
      const query = {
        bool: {
          must: [
            {
              term: {
                "experiment_id.keyword": experimentId
              }
            },
            {
              range: {
                "_seq_no": {
                  gt: lastSequenceId
                }
              }
            }
          ]
        }
      };

      const response = await this.client.search({
        index: this.index,
        body: {
          query: query,
          sort: [
            {
              "_seq_no": {
                order: "asc"
              }
            }
          ],
          size: size,
          _source: {
            includes: ["timestamp", "ch1", "ch2", "ch3", "ch4", "experiment_id"]
          }
        }
      });

      const hits = response.body.hits.hits;
      const newData = hits.map(hit => ({
        ...hit._source,
        sequenceId: hit._seq_no
      }));

      const lastSequenceIdFromResponse = hits.length > 0 
        ? Math.max(...hits.map(hit => hit._seq_no))
        : lastSequenceId;

      return {
        newData,
        lastSequenceId: lastSequenceIdFromResponse,
        totalHits: response.body.hits.total.value
      };
    } catch (error) {
      console.error('Elasticsearch query error:', error);
      throw new Error(`Failed to fetch data: ${error.message}`);
    }
  }

  async getExperimentInfo(experimentId) {
    try {
      const response = await this.client.search({
        index: this.index,
        body: {
          query: {
            term: {
              "experiment_id.keyword": experimentId
            }
          },
          size: 1,
          sort: [
            {
              "timestamp": {
                order: "desc"
              }
            }
          ]
        }
      });

      return {
        exists: response.body.hits.total.value > 0,
        totalDocuments: response.body.hits.total.value,
        latestTimestamp: response.body.hits.hits[0]?._source?.timestamp
      };
    } catch (error) {
      throw new Error(`Failed to get experiment info: ${error.message}`);
    }
  }

  async getAvailableExperiments() {
    try {
      const response = await this.client.search({
        index: this.index,
        body: {
          size: 0,
          aggs: {
            experiments: {
              terms: {
                field: "experiment_id.keyword",
                size: 100
              },
              aggs: {
                latest_timestamp: {
                  max: {
                    field: "timestamp"
                  }
                },
                doc_count: {
                  value_count: {
                    field: "timestamp"
                  }
                }
              }
            }
          }
        }
      });

      const buckets = response.body.aggregations.experiments.buckets;
      return buckets.map(bucket => ({
        experimentId: bucket.key,
        documentCount: bucket.doc_count,
        latestTimestamp: bucket.latest_timestamp.value
      }));
    } catch (error) {
      throw new Error(`Failed to get available experiments: ${error.message}`);
    }
  }

  async getIndexHealth() {
    try {
      const response = await this.client.indices.stats({
        index: this.index
      });

      return {
        exists: true,
        documentCount: response.body._all.primaries.docs.count,
        indexSize: response.body._all.primaries.store.size_in_bytes
      };
    } catch (error) {
      if (error.statusCode === 404) {
        return {
          exists: false,
          documentCount: 0,
          indexSize: 0
        };
      }
      throw new Error(`Failed to get index health: ${error.message}`);
    }
  }
}
